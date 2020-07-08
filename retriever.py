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
