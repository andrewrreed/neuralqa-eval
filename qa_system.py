import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers.data.metrics.squad_metrics import squad_evaluate

from reader import Reader
from retriever import Retriever


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s-%(levelname)s-%(message)s')


class QASystem:

    def __init__(self, retriever=Retriever(), reader=Reader()):

        self.retriever = retriever
        self.reader = reader

    def query(self, prediction_function, question, index_name, topN_docs, topM_passages, topk_answers, **kwargs):
        """
        Extracts answers from a full QA system: 
            1) Constructs query and retrieves the topk relevant documents 
            2) Passes those documents to the reader's prediction_function
            3) Returns the topk answers for each of the k documents

        Inputs:
            prediction_function: either 'predict' or 'predict_combine'
            question: str, question string
            index_name: str, name of the index for the retriever
            topk: int, number of documents to retrieve from retriever

        Outputs: 
            answers: dict with format:
                {
                'question': question string,
                'answers': list of answer dicts from reader 
                }
        """
        # Candidate Document Retrieval
        retriever_response, question_text = self.retriever.search_es(index_name=index_name, 
                                                                    question_text=question, 
                                                                    n_results=topN_docs,
                                                                    **kwargs)

        # Passage Ranking and Filtering
        pr = PassageRanker(question_text=question_text,
                           es_results=retriever_response['hits'],
                           n_passages=topM_passages)

        # Answer Prediction from Reader
        if prediction_function == 'predict_combined':
            pred_response = self.reader.predict_combined(question=question, 
                                                         documents=pr.output_docs,
                                                         topk_answers=topk_answers)
            preds = pred_response['preds']
        
        elif prediction_function == 'predict_simple':
            pred_response = self.reader.predict_simple(question=question, 
                                                    documents=pr.output_docs,
                                                    topk_answers=topk_answers)

            preds = pred_response['preds']


        return preds
    
        
    def evaluate_qa_sys(self, examples, prediction_function, index_name, full_sys, exp_name,
                        topN_docs, topM_passages, topk_answers, **kwargs):
        '''
        Takes in:
            1. prediction function
            2. List of question examples
            3. index name to search
            4. topN docs, topk_passages
            5. kwargs for pipeline settings (query expansion on/off, ranking on/off, etc)

        For each question example:
            - Expand Query
            - Query ES to get docs
            - Rerank passages
                - evaluate and save out retriever metrics 
            - Inference on concatentated passages
                - evaluate and save out reader metrics
            - aggregate save all reader & retriever metrics

        '''

        retriever_metrics = []
        ranker_metrics = []
        reader_metrics = []
        reader_predictions = {}

        for i, ex in enumerate(tqdm(examples)):

            if len(ex.answers) > 0:
                # Candidate Document Retrieval Evaluation
                retriever_response, question_text = self.retriever.search_es(index_name=index_name, 
                                                            question_text=ex.question_text,
                                                            n_results=topN_docs,
                                                            **kwargs)

                retriever_eval_data = self.calculate_nq_retriever_metrics(ex, retriever_response)
                retriever_metrics.append(retriever_eval_data)

                # Candidate Passage Retrieval Evaluation
                pr = PassageRanker(question_text=question_text,
                                    es_results=retriever_response['hits'],
                                    n_passages=topM_passages)

                ranker_eval_data = self.calculate_nq_ranker_metrics(ex, pr)
                ranker_metrics.append(ranker_eval_data)

            # Reader Comprehension Evaluation
            if full_sys:
                if prediction_function == 'predict_combined':

                    pred_response = self.reader.predict_combined(question=ex.question_text, 
                                                                documents=pr.output_docs,
                                                                topk_answers=topk_answers)


                    reader_predictions[ex.qas_id] = pred_response['preds'][0]['answer_text']
                    reader_metrics.append(pred_response['took'])

                elif prediction_function == 'predict_simple':

                    pred_response = self.reader.predict_simple(question=ex.question_text, 
                                                                documents=pr.output_docs,
                                                                topk_answers=topk_answers)


                    reader_predictions[ex.qas_id] = pred_response['preds'][0]['answer_text']
                    reader_metrics.append(pred_response['took'])

        # Aggregate metrics
        retriever_eval, retriever_df = self.aggregate_retrieval_metrics(retriever_metrics)
        ranker_eval, ranker_df = self.aggregate_retrieval_metrics(ranker_metrics)

        all_results = {'retriever_eval':retriever_eval,
                        'retriever_df':retriever_df,
                        'ranker_eval':ranker_eval,
                        'ranker_df':ranker_df}
                
        if full_sys:
            reader_eval = self.aggregate_reader_metrics(examples, reader_predictions, reader_metrics)
            all_results['reader_eval'] = reader_eval

        # Save eval settings
        exp_settings = {'exp_name': exp_name,
                        'prediction_function': prediction_function,
                        'index_name': index_name,
                        'full_sys': full_sys,
                        'exp_name': exp_name,
                        'topN_docs': topN_docs,
                        'topM_passages': topM_passages,
                        'topk_answers': topk_answers,
                        'entity_args': True if 'entity_args' in kwargs.keys() else False,
                        'synonym_args': (kwargs['synonym_args']['exp_type'], kwargs['synonym_args']['n_syns'])  if 'synonym_args' in kwargs.keys() else False
                    }
    
        all_results['exp_settings'] = exp_settings

        # save results
        if exp_name:
            filename = exp_name + '.pkl'
            with open(f'../data/evaluation_results/exp_2/{filename}', 'wb') as f:
                pickle.dump(all_results, f)

        return all_results
            
    @staticmethod
    def aggregate_reader_metrics(examples, reader_predictions, reader_metrics):
        '''
        Scores and aggregates reader metrics 

        '''

        reader_eval = dict(squad_evaluate(examples, reader_predictions))
        reader_took = np.mean(reader_metrics)
        reader_eval['Average Prediction Duration'] = reader_took

        return reader_eval


    @staticmethod
    def aggregate_retrieval_metrics(metrics_list):
        '''
        Aggregates retrieval metrics for the Retreiver and Ranker across all examples

        '''

        # format results dataframe
        cols = ['example_id', 'question', 'answer', 'duration', 'answer_present', 'average_precision']
        results_df = pd.DataFrame(metrics_list, columns=cols)

        # format results dict
        metrics = {'Recall': 1-results_df.answer_present.value_counts(normalize=True)[0],
                   'Mean Average Precision': results_df.average_precision.mean(),
                   'Average Query Duration':results_df.duration.mean()}

        return metrics, results_df


    def calculate_nq_retriever_metrics(self, example, retriever_response):
        '''
        Calculate performance metrics from query search_es() response
        
        '''
        duration = retriever_response['took']
        example_answers = [ex['text'] for ex in example.answers]
        
        # check for presence of any answer in each document
        binary_results = []
        for doc in retriever_response['hits']:
            doc_res = []
            for ans in example_answers:
                doc_res.append(ans.lower() in doc.text.lower())
            binary_results.append(int(any(doc_res)))
            
        ans_in_res = int(any(binary_results))
        
        # calculate average precision
        ap = self.average_precision(binary_results)
        
        rec = (example.qas_id, example.question_text, example_answers, duration, ans_in_res, ap)

        return rec

    def calculate_nq_ranker_metrics(self, example, passage_ranker):
        '''
        Calculate performance metrics from PassageRankerresponse
        
        '''

        duration = passage_ranker.took
        example_answers = [ex['text'] for ex in example.answers]

        # check for presence of any answer in each document
        binary_results = []
        for doc in passage_ranker.output_docs:
            doc_res = []
            for ans in example_answers:
                doc_res.append(ans.lower() in doc['text'].lower())
            binary_results.append(int(any(doc_res)))

        ans_in_res = int(any(binary_results))

        # calculate average precision
        ap = self.average_precision(binary_results)

        rec = (example.qas_id, example.question_text, example_answers, duration, ans_in_res, ap)

        return rec


    @staticmethod
    def average_precision(binary_results):
        
        ''' Calculates the average precision for a list of binary indicators '''
        
        m = 0
        precs = []

        for i, val in enumerate(binary_results):
            if val == 1:
                m += 1
                precs.append(sum(binary_results[:i+1])/(i+1))
                
        ap = (1/m)*np.sum(precs) if m else 0
                
        return ap




class PassageRanker:
    '''
    Passage ranking utility class that optionally:
        1. Takes in search results from Elasticsearch-dsl
        2. Parses out each paragraph as a passage
        3. Builds sparse TF-IDF vectors for each passage + the input question
        4. Computes cosine similarity for each passage vs. question vector
        5. Saves most similar passage records

    Note - If n_passages is None, no chunking or ranking is performed and whole documents
            are saved as is.

    Args:
        question_text (str)
        ex_restuls (elasticsearch_dsl.utils.AttrList)
        n_passages (int) - number of most similar passages save

    '''
    
    def __init__(self, question_text, es_results, n_passages):
        
        self.start_time = time.time()
        self.question_text = question_text
        self.docs, self.passages_df = self.format_input(es_results)
        self.n_passages = n_passages
        
        if self.n_passages:
            self.calculate_tfidf_similarity()
            self.format_passage_output()
        else:
            self.format_document_output()
        
        self.end_time = time.time()
        self.took = (self.end_time - self.start_time)*1000 #to milliseconds

    
    def format_document_output(self):
        '''
        If no passage ranking is desired, simply save all full docs for processing
        by the Reader

        '''
        self.output_docs = [{k:v for k, v in rec.items() if k in ['id', 'title', 'text']} \
                            for rec in self.docs]
    
    def format_passage_output(self):
        '''
        Takes the top N most similar passages and formats them for processing
        by the Reader.
        
        '''
        most_sim = self.passages_df.sort_values('sim_score', ascending=False)\
                    .reset_index(drop=True)\
                    .loc[:self.n_passages-1, ['id', 'title', 'text']]
        
        self.output_docs = most_sim.to_dict(orient='records')
        
        
    def calculate_tfidf_similarity(self):
        '''
        Uses the passages_df to vectorize each passage and the question, then
        computes cosine similarity of those sparse vectors and saves them back into
        passages_df attribute
        
        '''
        vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
        
        combined = self.passages_df['text'].append(pd.Series(self.question_text),
                                                           ignore_index=True)

        passage_vecs, question_vec = np.split(vectorizer.fit_transform(combined).toarray(),[-1])
        
        sim_vec = cosine_similarity(passage_vecs, question_vec)

        self.passages_df['sim_score']  = sim_vec.flatten()
        
        
    @staticmethod
    def format_input(es_results):
        '''
        Take raw results from Elasticsearch-dsl and format as Dataframe
        
        '''
        # format as dicts
        docs = []
        for rec in es_results:
            doc = {
                'id': rec.meta.id,
                'score': rec.meta.score,
                'text': rec.text, 
                'title': rec.title}
            
            doc['passages'] = doc['text'].split('\n\n')
            docs.append(doc)
        
        # formate as dataframe
        passages_recs = []
        
        for doc in docs:
            for i, passage in enumerate(doc['passages']):
                
                if i == 0 or passage in ['\n', '', ' ']:
                    # skip article title
                    continue
                    
                rec = (doc['id'], doc['title'], i, passage)
                passages_recs.append(rec)
                
        passages_df = pd.DataFrame(data=passages_recs, columns=['id', 'title', 'passage_index', 'text'])
        
        return docs, passages_df
