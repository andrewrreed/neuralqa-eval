import json
import os
import sys
import logging
import re
import pickle
from tqdm import tqdm
from utils.text_utils import simplify_nq_example

def convert_nq_dev_to_squad_format(filepath):
    '''
    Load NQ dev set from disk, simplify each record, convert them to SQuAD format
    
    '''
    
    nq_examples = []
    yes_no_count = 0
    with open(filepath, 'rb') as f:
        for i, line in enumerate(tqdm(f)):

            simp_example = simplify_nq_example(json.loads(line.decode('utf-8')))
            answers, yes_no_flag = get_short_answers_from_span(simp_example)

            if yes_no_flag:
                # exclude questions with any annotation indicating yes/no question
                yes_no_count += 1
                continue

            clean_record = {'qas_id': simp_example['example_id'],
                            'title': extract_wiki_title(simp_example['document_url']),
                            'question_text': simp_example['question_text'],
                            'answers': answers,
                            'is_impossible': True if len(answers)==0 else False}
            
            nq_ex = NQSquadExample(**clean_record)

            nq_examples.append(nq_ex)
    
    print(f'Found and removed {yes_no_count} yes/no questions. {len(nq_examples)} remain.')
            
    return nq_examples

def get_short_answers_from_span(simplified_example):
    '''
    Extracts short answer text from a simplified NQ example using the short answer span and document text and
    returns flag if any annotation indicates a yes/no answer
    
    Note:
        1. Annotations that have multipart answers (more than 1 short answer) are dropped from list
            of short answers
        2. Answers with many tokens often resemble extractive snippets rather than canonical answers, 
            so we discard answers with more than 5 tokens. (https://arxiv.org/pdf/1906.00300.pdf)
    
    '''
    
    answers = []
    yes_no_flag = False
    for annotation in simplified_example['annotations']:
        
        # check for yes/no questions
        if annotation['yes_no_answer'] != 'NONE':
            yes_no_flag = True
        
        # extract short answers
        if len(annotation['short_answers']) > 1 or len(annotation['short_answers']) == 0:
            continue
                
        else:
            short_answer_span = annotation['short_answers'][0]
            short_answer = " ".join(simplified_example['document_text'].split(" ")\
                                    [short_answer_span['start_token']:short_answer_span['end_token']])
            
            if len(short_answer.split(' ')) > 5:
                continue
            
            answers.append(short_answer)
            
    return answers, yes_no_flag

def extract_wiki_title(document_url):
        '''
        This function applies a regular expression to an input wikipedia article URL
        to extract and return the article title.
        
        Args:
            document_url (string)
            
        Returns:
            title (string) - article title
        '''
        
        pattern = 'title=(.*?)&amp'
        
        try:
            title = re.search(pattern, document_url).group(1)
        except AttributeError:
            title = 'No Title Found'
            
        return title

class NQSquadExample(object):
    """
    A single dev example for the NQ dataset represented in SQuAD format
    
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        title: The title of the Wikipedia article
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        title,
        answers,
        is_impossible,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
