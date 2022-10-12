import pandas as pd
from math import log
import operator
data = pd.read_table('spam.csv', encoding='latin-1', header=None)
df = pd.DataFrame(data=(s.split(',', 1) for s in data[0][1:]), columns=['class','text'])
df['class_id'] = df['class']=='spam'

sents = []
labels = []
data = []
ALL = []
cl_words = dict((key, []) for key in df.class_id)
for i in xrange(0, int(len(df) * 0.8) ):
    words = df.text[i].split()
    sents.append(words)
    labels.append(df.class_id[i])
    ALL += words
    cl_words[df.class_id[i]] += words

# calculate P(c)
priors = {}
for cl in labels:
    priors[cl] = log ( float(labels.count(cl)) / len(labels) )

vocab = set(ALL)

# likelihood P(w|c):   word : { class1: log , class2: log }
mles = dict((key, {}) for key in vocab)
for word in vocab:
    for cl in cl_words.keys():
        mle = log ( (cl_words[cl].count(word) + 1) / float(( ALL.count(word) + len(vocab) )) )
        mles[word][cl] = mle

# test_sms = "Nah I don't think he goes to usf, he lives around here though"
# Naive Bayes
def NaiveBayes(cl_words, priors, mles, sent):
    preds = {}
    for cl in cl_words.keys():
        pred = priors[cl]
        for word in sent:
            pred += mles[word][cl]
        preds[cl] = pred
    return max(preds.iteritems(), key=operator.itemgetter(1))[0]

# print NaiveBayes(cl_words, priors, mles, test_sms)

true_labels = []
pred_labels = []

for i in xrange(int(len(df) * 0.8), int(len(df))):
    words = df.text[i].split()
    true_l = df.class_id[i]
    pred_l = NaiveBayes(cl_words, priors, mles, words)
    true_labels.append(true_l)
    pred_labels.append(pred_l)

tp = 0
fp = 0
fn = 0
for i in xrange(0, len(true_labels)):
    if true_labels[i] == pred_labels[i]:
        tp += 1
    elif true_labels[i] == True and pred_labels[i] == False:
        fp += 1
    elif true_labels[i] == False and pred_labels[i] == True:
        fn += 1

precision = float(tp) / ( tp + fp )
recall = float(tp) / ( tp + fn )
f1 = 2 * ( ( precision * recall ) / ( precision + recall ) )

print f1