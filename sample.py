# -*- coding: utf-8 -*-

import re
from nltk.corpus import movie_reviews, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
import numpy as np

#==== General parameters
FEATURES_NUMBER = 2000
NGRAMS_NUMBER = 2
REGULARISATION = 10.0

#==== Gradient descent constants
SPEED = 0.001
MAX_ITERATIONS = 20
THRESHOLD_CONVERGENCE = 1 # in percentage

#==== Text processing constants
BLACKLIST_STOPWORDS = ['over','only','very','not','no']
ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)
NEG_CONTRACTIONS = [
    (r'aren\'t', 'are not'),
    (r'can\'t', 'can not'),
    (r'couldn\'t', 'could not'),
    (r'daren\'t', 'dare not'),
    (r'didn\'t', 'did not'),
    (r'doesn\'t', 'does not'),
    (r'don\'t', 'do not'),
    (r'isn\'t', 'is not'),
    (r'hasn\'t', 'has not'),
    (r'haven\'t', 'have not'),
    (r'hadn\'t', 'had not'),
    (r'mayn\'t', 'may not'),
    (r'mightn\'t', 'might not'),
    (r'mustn\'t', 'must not'),
    (r'needn\'t', 'need not'),
    (r'oughtn\'t', 'ought not'),
    (r'shan\'t', 'shall not'),
    (r'shouldn\'t', 'should not'),
    (r'wasn\'t', 'was not'),
    (r'weren\'t', 'were not'),
    (r'won\'t', 'will not'),
    (r'wouldn\'t', 'would not'),
    (r'ain\'t', 'am not') # not only but stopword anyway
]

OTHER_CONTRACTIONS = {
    "'m": 'am',
    "'ll": 'will',
    "'s": 'has', # or 'is' but both are stopwords
    "'d": 'had'  # or 'would' but both are stopwords
}