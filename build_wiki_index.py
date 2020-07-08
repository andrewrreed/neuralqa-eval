################################################################################################
# Script to download and create a local Elasticsearch index for the latest full Wikipedia dump

# Note - This script assumes Elasticsearch is running locally on port 9200

################################################################################################

import os
from retriever import Retriever

# download WikiExtractor
os.system("curl -O https://raw.githubusercontent.com/attardi/wikiextractor/master/WikiExtractor.py")

# get latest Wikipedia dump
if not os.path.exists('data/'):
    os.makedirs('data/')

os.system('cd data && wget http://download.wikimedia.org/wiki/latest/wiki-latest-pages-articles.xml.bz2')

# extract Wikipedia dump
os.system("python WikiExtractor.py -o data/. --json enwiki-latest-pages-articles.xml.bz2")

## build Elasticsearch index
ret = Retriever()

# create new index
index_config = {
    "settings": {
        "analysis": {
            "analyzer": {
                "stop_stem_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter":[
                        "lowercase",
                        "stop",
                        "snowball"
                    ]
                    
                }
            }
        }
    },
    "mappings": {
        "dynamic": "strict", 
        "properties": {
            "id": {"type": "integer"},
            "title": {"type": "text", "analyzer": "stop_stem_analyzer"},
            "text": {"type": "text", "analyzer": "stop_stem_analyzer"},
            "url": {"type": "text"}
            }
        }
    }
ret.create_es_index(index_name='wiki-full-stop-stem', index_config=index_config)

# populate index
ret.populate_wiki_index(index_name='wiki-full-stop-stem', data_dir='data/')