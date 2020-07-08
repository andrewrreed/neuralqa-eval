# Question Answering Evaluation Framework

In a open domain question answering system, a two-stage approach is often employed to a.) search a large corpus and retrieve candidate documents based on an input query and b.) apply a reading comprehension algorithm to process the candidate documents and extract a specific answer that best satisfies the input query. 

This repo provides a framework for evaluating an end-to-end question answering system on the [Natural Questions](https://ai.google.com/research/NaturalQuestions/) (NQ) dataset. The framework consists of several parts:

1. Automated data preprocessing utilities for the NQ development dataset
2. Automated ElasticSearch setup to enable candidate document retrieval on a full Wikipedia dump
3. Query expansion and passage ranking utilities for QA system optimizaion
3. A pre-trained BERT model machine reading comprehension
5. End-to-end tools for evaluating the QA system 

### The Dataset

Google's [Natural Questions](https://ai.google.com/research/NaturalQuestions/) corpus - a question answering dataset - consists of real, anonymized, aggregated queries issued to the Google search engine that are paired with high quality, manual annotations. The nature by which question/answer examples are created presents a unique challenge compared to previous question answering datasets making solutions to this task much more representative of true open-domain question answering systems.

### Document Retriever

ElasticSearch - a distributed, open source search engine built on Apache Lucene - is used in the framework for retrieving candidate documents. ElasticSearch utilizes the BM25 algorithm for information retrieval based on exact keyword matches and offers an extensive and easy to used API for search setup and interaction.

### Document Reader

Transformer based reading comprehension approaches have started to outperform humans on some key NLP benchmarks including question answering. Specifically, we utilize versions of BERT pre-trained on SQuAD2.0 to understand how well that find-tuned model performs on the more difficult NQ dataset.
