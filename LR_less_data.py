import pandas as pd
from math import log
from math import exp
import operator

data = pd.read_table('spam.csv', encoding='latin-1', header=None)
df = pd.DataFrame(data=(s.split(',', 1) for s in data[0][1:]), columns=['class', 'text'])
df['class_id'] = df['class'] == 'spam'

def prep_train_data(df):
    sents = []
    labels = []
    ALL = []

    cl_words = dict((key, []) for key in df.class_id)

    for i in xrange(0, int(len(df) * 0.8)):
        words = df.text[i].split()
        sents.append(words)
        labels.append(df.class_id[i])
        ALL += words
        cl_words[df.class_id[i]] += words

    vocab = set(ALL)

    return sents, labels, ALL, vocab, cl_words

def priors(labels):
    priors = {}
    for cl in labels:
        priors[cl] = log(float(labels.count(cl)) / len(labels))
    return priors

def likelihood(vocab, cl_words):
    mles = dict((key, {}) for key in vocab)
    for word in vocab:
        for cl in cl_words.keys():
            mle = log((cl_words[cl].count(word) + 1) / float((ALL.count(word) + len(vocab))))
            mles[word][cl] = mle
    return mles

def NaiveBayes(cl_words, priors, mles, sent):
    preds = {}
    for cl in cl_words.keys():
        pred = priors[cl]
        for word in sent:
            if word in mles.keys():
                pred += mles[word][cl]
        preds[cl] = pred
    return max(preds.iteritems(), key=operator.itemgetter(1))[0]

def LogisticRegression(cl_words, mles, sent):
    preds = {}
    for cl in cl_words.keys():
        wf = 0
        for word in sent:
            if word in mles.keys():
                wf += mles[word][cl]
        p_c = exp( exp(wf) )
        preds[cl] = p_c
    sum_p_c = sum(preds.values())
    for cl in preds:
        preds[cl] = preds[cl] / sum_p_c
    return max(preds.iteritems(), key=operator.itemgetter(1))[0]

def calc_F1(true_labels, pred_labels):
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

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return f1

sents, labels, ALL, vocab, cl_words = prep_train_data(df)
priors = priors(labels)
mles = likelihood(vocab, cl_words)

true_labels = []
NB_pred_labels = []
LR_pred_labels = []
for i in xrange(int(len(df) * 0.8), int(len(df))):
    words = df.text[i].split()
    true_l = df.class_id[i]
    NB_pred_l = NaiveBayes(cl_words, priors, mles, words)
    LR_pred_l = LogisticRegression(cl_words, mles, words)
    true_labels.append(true_l)
    NB_pred_labels.append(NB_pred_l)
    LR_pred_labels.append(LR_pred_l)

NB_F1 = calc_F1(true_labels, NB_pred_labels)
LR_F1 = calc_F1(true_labels, LR_pred_labels)
print "F1 for Naive Bayes + Laplace smoothing: ", NB_F1
print "F1 for Logistic Regression: ", LR_F1