####################################
#  Class and utils for the Reader  #
####################################
import os
import time
from transformers import pipeline

class Reader:
    def __init__(
        self, 
        model_name="twmkn9/distilbert-base-uncased-squad2",
        tokenizer_name="twmkn9/distilbert-base-uncased-squad2",
        use_gpu=True,
        handle_impossible_answer=True,
        relative_softmax=True
        ):
        
        self.use_gpu = use_gpu
        self.model = pipeline('question-answering', model=model_name, tokenizer=tokenizer_name, device=int(use_gpu)-1)
        self.kwargs = {'handle_impossible_answer':handle_impossible_answer}

    def predict_combined(self, question, documents, topk_answers=1):
        """
        Compute text prediction for a question given a collection of documents

        Inputs:
            question: str, question string
            documents: list of document dicts, each with the following format:
                    {
                        'text': context string,
                        'id': document identification,
                        'title': name of document
                    }
            topk (optional): int, if provided, overrides default topk
        
        Outputs:
            results: dict with the following format:
                {
                    'question': str, question string,
                    'answers': list of answer dicts, each including text answer, probability, 
                                start and end positions, and document metadata
                }
        """

        start_time = time.time()
        self.kwargs['topk'] = topk_answers

        # combine all documents together into one long context
        context = ''
        for doc in documents:
            context += doc['text'] + ' '

        print(f'Concatenated {len(documents)} docs for reading.')

        # make predictions
        all_predictions = []
        inputs = {"question": question, "context": context}
        predictions = self.model(inputs, **self.kwargs)
        

        if self.kwargs['topk'] == 1:
            predictions = [predictions]

        for pred in predictions:
            answer = {
                    'probability': pred['score'],
                    'answer_text': pred['answer'],
                    'start_index': pred['start'],
                    'end_index': pred['end'],
                    'id': doc['id'],
                    'title': doc['title']
                }
            all_predictions.append(answer)

        # sort and truncate predictions
        preds = sorted(all_predictions, key=lambda x: x["probability"], reverse=True)[: self.kwargs["topk"]]
        
        end_time = time.time()
        took = (end_time - start_time)*1000 #to milliseconds
        
        pred_response = {'preds':preds,
                         'took': took}

        return pred_response


    def predict_simple(self, question, documents, topk_answers=1):
        '''
        Generates a prediction for each doc in documents, then takes the best k as scored individually by the
        Reader.

        Note - Currently, this is not a fair comparison across documents because we are comparing probabilities
                and not the output logits themselves...TO-DO...fix this

        '''
        start_time = time.time()
        # manually set model options to only take the best one answer from each passage
        self.kwargs['topk'] = 1

        all_predictions = []
        for doc in documents:        
            inputs = {"question": question, "context": doc['text']}
            predictions = self.model(inputs, **self.kwargs)
            best = predictions

            answer = {
                    'probability': best['score'],
                    'answer_text': best['answer'],
                    'start_index': best['start'],
                    'end_index': best['end'],
                    'doc_id': doc['id'],
                    'title': doc['title']
                }
            all_predictions.append(answer)

        # Simple heuristic:
        # If the best prediction from each document is the null answer, return null
        # Otherwise, return the highest scored non-null answer
        null = True
        for prediction in all_predictions:
            if prediction['answer_text']:
                null = False
        
        if not null:
            # pull out and sort only non-null answers
            non_null_predictions = [prediction for prediction in all_predictions if prediction['answer_text']]
            sorted_non_null = sorted(non_null_predictions, key=lambda x: x['probability'], reverse=True)
            
            # append the null answers for completeness
            null_predictions = [prediction for prediction in all_predictions if not prediction['answer_text']]
            best_predictions = sorted_non_null + null_predictions
        else:  
            # sort null answers for funsies
            best_predictions = sorted(all_predictions, key=lambda x: x["probability"], reverse=True)
        
        end_time = time.time()
        took = (end_time - start_time)*1000 #to milliseconds

        results = {'preds': best_predictions[: self.kwargs['topk']],
                   'took': took
                   }

        return results