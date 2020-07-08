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
    
        
