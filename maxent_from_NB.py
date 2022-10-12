from os import listdir
from math import log
from math import exp
import operator
import itertools



def prep_data(DIR, labels):
    sents = []
    ALL = []
    all_labels = []

    cl_words = dict((key, []) for key in labels)

    for class_id in labels:
        for file in listdir(DIR + class_id):
            fh = open(DIR + class_id + '/' + file, 'r')
            text = fh.read()
            words = text.split()
            ALL += words
            sents.append(words)
            cl_words[class_id] += words
            all_labels.append(class_id)

    vocab = set(ALL)

    return sents, ALL, vocab, cl_words, all_labels

def likelihood(ALL, vocab, cl_words):
    mles = dict((key, {}) for key in vocab)
    for word in vocab:
        for cl in cl_words.keys():
            mle = log((cl_words[cl].count(word) + 1) / float(ALL.count(word) + len(vocab)))
            mles[word][cl] = mle
    print 'likelihood done'
    return mles

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

#vocab = open('aclImdb/imdb.vocab', 'r').read().split('\n')
vocab = []
POS_words = []
NEG_words = []
NEG_sents = []
POS_sents = open('aclImdb/train-pos.txt', 'r').read().split('\n')

for sent in POS_sents:
    sent = sent.split()
    POS_words += sent
    sent = set(sent)
    vocab += sent

for sent in NEG_sents:
    sent = sent.split()
    NEG_words += sent
    sent = set(sent)
    vocab += sent

NEG_sents = open('aclImdb/train-neg.txt', 'r').read().split('\n')
cl_words = {'POS': POS_words, 'NEG': NEG_words}
ALL = NEG_words + POS_words
mles = likelihood(ALL, vocab, cl_words)
# test_POS = open('aclImdb/test-pos.txt', 'r').read().split('\n')
# test_NEG = open('aclImdb/test-neg.txt', 'r').read().split('\n')
# test_all = {'POS': test_POS, 'NEG': test_NEG}

# pred_labels = []
# true_labels = []
# for cl in test_all:
#     for sent in cl:
#         pred_l = LogisticRegression(cl_words, mles, sent)
#         pred_labels.append(pred_l)
#         true_labels.append(cl)

# F1 = calc_F1(true_labels, pred_labels)
# print F1
print mles