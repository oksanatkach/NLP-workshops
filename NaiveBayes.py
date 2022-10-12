import pandas as pd
from math import log
import operator
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

'''
# data prep
ed = open('lohika_set/education.txt', 'r').read().decode('utf-8')
exp = open('lohika_set/experience.txt', 'r').read().decode('utf-8')
tech = open('lohika_set/technology.txt', 'r').read().decode('utf-8')

def normalize(raw):
    minlength = 2
    raw = raw.lower()
    lst = nltk.Text(word_tokenize(raw))

    clean = [token for token in lst if
                        (not token in stopwords.words('english')) and len(token) > minlength]

    lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

    for ind in range(len(clean)):
        token = clean[ind]
        clean[ind] = lmtzr.lemmatize(token)

    return clean

ed = normalize(ed)
exp = normalize(exp)
tech = normalize(tech)

all_dict = {'ed':ed, 'exp':exp, 'tech':tech}
all_len = len(ed) + len(exp) + len(tech)
ALL = list(set(ed + exp + tech))

# Naive Bayes
def priors(labels, all_len):
    priors = {}
    for cl in labels:
        # priors[cl] = log(float(len(labels[cl])) / all_len)
        priors[cl] = float(len(labels[cl]) / float(all_len))
    return priors

'''

data = pd.read_table('spam.csv', encoding='latin-1', header=None)
df = pd.DataFrame(data=(s.split(',', 1) for s in data[0][1:]), columns=['class', 'text'])
df['class_id'] = df['class'] == 'spam'

print df


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

priors = priors(all_dict, all_len)
mles = likelihood(ALL, all_dict)


def NaiveBayes(cl_words, priors, mles, sent):
    preds = {}
    for cl in cl_words.keys():
        pred = priors[cl]
        for word in sent:
            if word in mles.keys():
                pred += mles[word][cl]
        preds[cl] = pred
    return max(preds.iteritems(), key=operator.itemgetter(1))[0]


print NaiveBayes(all_dict, priors, mles, ['national', 'university'])


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
pred_labels = []
for i in xrange(int(len(df) * 0.8), int(len(df))):
    words = df.text[i].split()
    true_l = df.class_id[i]
    pred_l = NaiveBayes(cl_words, priors, mles, words)
    print pred_l
    true_labels.append(true_l)
    pred_labels.append(pred_l)

F1 = calc_F1(true_labels, pred_labels)
print F1
