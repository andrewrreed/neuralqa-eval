import re
import os
import json
import pickle as pkl
import logging
import time
from tqdm import tqdm
import spacy
import logging
import en_core_web_lg
from elasticsearch_dsl import Q, Search
from elasticsearch import Elasticsearch
import gensim.downloader as api


class Retriever:

    def __init__(self, es_config={'host':'localhost', 'port':9200}):

        self.es_config = es_config
        self.es = self.connect_es(**es_config)
        self.nlp = en_core_web_lg.load()
        self.word_vectors = api.load("glove-wiki-gigaword-100")

    def connect_es(self, host='localhost', port=9200):
        '''
        Instantiate and return a Python ElasticSearch object

        Args:
            host (str)
            port (int)
        
        Returns:
            es (elasticsearch.client.Elasticsearch)

        '''
        
        config = {'host':host, 'port':port}

        try:
            es = Elasticsearch([config])

        except Exception as e:
            logging.error('Couldnt connect to ES server', exc_info=e)

        return es

    def create_es_index(self, index_name, index_config):
        '''
        Create an ElasticSearch index

        Args:
            es_obj (elasticsearch.client.Elasticsearch)
            settings (dict)
            index_name (str)

        '''

        self.es.indices.create(index=index_name, body=index_config, ignore=400)

        logging.info('Index created successfully!')

        return

    def populate_wiki_index(self, index_name, data_dir):
        '''
        Loads records into an existing Elasticsearch index from Wikiextractor JSON dump

        Args:
            es_obj (elasticsearch.client.Elasticsearch) - Elasticsearch client object
            index_name (str) - Name of index
            data_dir (str) - path to root directory of wiki JSON dump

        '''
        
        files = []
        for dirname, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                files.append(os.path.join(dirname, filename))
        
        
        for file in tqdm(files):
            
            data = []
            with open(file, 'rb') as f:
                for line in f:
                    data.append(json.loads(line))

            for i, rec in enumerate(data):
                try:
                    index_status = self.es.index(index=index_name, body=rec)
                except:
                    print(f'Unable to load document: {file, i, rec["title"]}.')
                
        time.sleep(5)
        n_records = es_obj.count(index=index_name)['count']
        print(f'Succesfully loaded {n_records} into {index_name}')

        return

    def search_es(self, index_name, question_text, n_results, **kwargs):
        '''
        Execute an Elasticsearch query on a specified index
        
        Args:
            es_obj (elasticsearch.client.Elasticsearch) - Elasticsearch client object
            index_name (str) - Name of index to query
            n_results (int) - Number of results to return
            expansion_args (dict) - entity and synonym arguments for QueryExpander class
                - 'ner_exp': True
                - 'synonym_exp': {'exp_type':'static, 'n_syns':3}
            
        Returns
            res - Elasticsearch response object
        
        '''
        # Note this was done so the word_vectors didn't have to reload with each search call
        local_exp_args = {}

        if 'entity_args' in kwargs.keys():
            ent_args = {'spacy_model': self.nlp} if kwargs['entity_args']['ner_exp'] == True else None
            local_exp_args['entity_args']=ent_args
        
        if 'synonym_args' in kwargs.keys():
            syn_args = {'gensim_model': self.word_vectors, 'n_syns': kwargs['synonym_args']['n_syns']} if kwargs['synonym_args'] else None
            local_exp_args['synonym_args']=syn_args

        # construct query
        qe = QueryExpander(question_text, **local_exp_args)
        query= qe.query

        # execute search
        s = Search(using=self.es, index=index_name)
        res = s.query(query)[:n_results].execute()

        # format results
        results = {'hits':res,
                    'took':res.took}

        question_text = qe.expanded_question if hasattr(qe, 'expanded_question') else question_text
        
        return results, question_text


class QueryExpander:
    '''
    Query expansion utility class that augments an ElasticSearch query with optional techniques
    including Named Entity Recognition and Synonym Expansion
    
    Args:
        question_text
        entity_args (dict) - Ex. {'spacy_model': nlp}
        synonym_args (dict) - Ex. {'gensim_model': word_vectors, 'n_syns': 3}
    
    '''
    
    def __init__(self, question_text, entity_args=None, synonym_args=None):
        
        self.question_text = question_text
        self.entity_args = entity_args
        self.synonym_args = synonym_args

        if self.synonym_args and not self.entity_args:
            raise Exception('Cannot do synonym expansion without NER! Expanding synonyms on named entities reduces recall.')

        if self.synonym_args or self.entity_args:
            self.nlp = self.entity_args['spacy_model']
            self.doc = self.nlp(self.question_text)
        
        self.build_query()
        
    def build_query(self):

        # build entity sub-query
        if self.entity_args:
            self.extract_entities()
        
        # identify terms to expand
        if self.synonym_args:
            self.identify_terms_to_expand()
        
        # build question sub-query
        self.construct_question_query()
        
        # combine sub-queries
        sub_queries = []
        sub_queries.append(self.question_sub_query)
        if hasattr(self, 'entity_sub_queries'):
            sub_queries.extend(self.entity_sub_queries)
            
        query = Q('bool', should=[*sub_queries])
        self.query = query
        
    
    def extract_entities(self):
        '''
        Extracts named entities using spaCy and constructs phrase match sub-queries
        for each. Saves both entities and sub-queries as attributes.
        
        '''
        
        entity_list = [entity.text.lower() for entity in self.doc.ents]
        
        entity_sub_queries = []
        
        for ent in entity_list:
            eq = Q('multi_match',
                   query=ent,
                   type='phrase',
                   fields=['title', 'text'])
            
            entity_sub_queries.append(eq)
        
        self.entities = entity_list
        self.entity_sub_queries = entity_sub_queries
        
        
    def identify_terms_to_expand(self):
        '''
        
        '''
        if hasattr(self, 'entities'):
            # get unique list of entity tokens
            entity_terms = [ent.split(' ') for ent in self.entities]
            entity_terms = [ent for sublist in entity_terms for ent in sublist]
        else:
            entity_terms = []
    
        # terms to expand not part of entity or a stopword
        terms_to_expand = [term for term in self.doc if \
                           (term.lower_ not in entity_terms) and (not term.is_stop)\
                            and (not term.is_digit) and (not term.is_punct) and 
                            (not len(term.lower_)==1 and term.is_alpha)]
        
        self.terms_to_expand = terms_to_expand

        
    def construct_question_query(self):
        '''
        Builds a multi-match query from the raw question text extended with synonyms 
        for eligible any terms (i.e. terms that are not part of an entity or stopword)

        '''
        
        if hasattr(self, 'terms_to_expand'):
            
            syns = []
            for term in self.terms_to_expand:
                syns.extend(self.gather_synonyms_static(term, self.synonym_args['n_syns']))
            
            question = self.question_text + ' ' + ' '.join(syns)
            self.expanded_question = question
            self.all_syns = syns
        
        else:
            question = self.question_text
        
        qq = Q('multi_match',
               query=question,
               type='most_fields',
               fields=['title', 'text'])
        
        self.question_sub_query = qq
        

    def gather_synonyms_static(self, token, n_syns):
        '''
        Takes in a token and returns a specified number of synonyms defined by
        cosine similarity of word vectors. Uses stemming to ensure none of the
        returned synonymns share the same stem (ex. photo and photos cant happen)
        
        '''
        try:
            syns = self.synonym_args['gensim_model'].similar_by_word(token.lower_)

            lemmas = []
            final_terms = []
            for item in syns:
                term = item[0]
                lemma = self.nlp(term)[0].lemma_

                if lemma in lemmas:
                    continue
                else:
                    lemmas.append(lemma)
                    final_terms.append(term)
                    if len(final_terms) == n_syns:
                        break
        except:
            final_terms = []

        return final_terms