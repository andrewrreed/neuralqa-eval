################################################################################################
# Script to create a SQuAD-like dataset from NQ dev set

# Note - This script you have downloaded and unzipped the NQ dev set to the data/nq/ directory

################################################################################################

import json
import os
import sys
import logging
import re
import pickle
from tqdm import tqdm

# download Google's provided script to simplify the dev dataset
if not os.path.exists('utils/text_utils.py'):
    os.system("cd utils && curl -O https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/text_utils.py")

from utils.data_utils import convert_nq_dev_to_squad_format

jsonfilename = "data/nq/v1.0-simplified_nq-dev-all.jsonl"
nq_examples = convert_nq_dev_to_squad_format(jsonfilename)

with open('data/nq/squad_format_nq_dev.pkl', 'wb') as f:
    pickle.dump(nq_examples, f)