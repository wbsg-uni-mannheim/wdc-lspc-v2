import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
from copy import deepcopy
import string

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def tokenize(words):
    #check for NaN
    if isinstance(words, float):
        if words != words:
            return []
    words = str(words)
    words = words.replace('&amp;', '')
    words = words.replace('&reg;', '')
    words = words.replace('&quot;', '')
    words = words.replace('\t;', ' ')
    words = words.replace('\n;', ' ')
    return words.lower().translate(str.maketrans('', '', string.punctuation)).split()

def preprocess_string(words, stop_words):
    #check for NaN
    if isinstance(words, float):
        if words != words:
            return words
    
    word_list = tokenize(words)
    word_list_stopwords_removed = [x for x in word_list if x not in stop_words]
    words_processed = ' '.join(word_list_stopwords_removed)
    return words_processed

def preprocess_string_column(column):
    stop_words_with_punct = deepcopy(stopwords.words('english'))
    stop_words = list(map(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)), stop_words_with_punct))
    
    column = column.apply(preprocess_string, args=(stop_words,))
        
    return column
