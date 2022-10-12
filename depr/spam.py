import pandas as pd
import numpy as np
from nltk import NaiveBayesClassifier
df = pd.read_csv('spam.csv', sep=',', usecols=[0, 1])
ALL = []
for i in xrange(0, len(df)):
    df.v2[i] = df.v2[i].split()
    ALL.append(df.v2[i])


# flat = [item for sublist in ALL for item in sublist]
# vocab = list(set(flat))
#
# for i in xrange(0, len(df)):
#     vector = [0] * len(vocab)
#     for word in vocab:
#         if word in df.v2[i]:
#             vector[vocab.index(word)] = 1


# dict = {'ham':[], 'spam':[]}
# vectors = {'ham':[], 'spam':[]}
#
# for i in xrange(0, len(df)):
#     if  df.v1[i] == 'spam':
#         dict['spam'].append(df.v2[i])
#     else:
#         dict['ham'].append(df.v2[i])
#
# ALL = dict['spam'] + dict['ham']
# flat = [item for sublist in ALL for item in sublist]
# vocab = list(set(flat))
#
# for key in dict:
#     for sent in dict[key]:
#         vector = [0] * len(vocab)
#         for word in vocab:
#             if word in sent:
#                 vector[vocab.index(word)] = 1
#         vectors[key].append(vector)