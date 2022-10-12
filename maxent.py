from os import listdir
from sklearn.linear_model import LogisticRegression

def read_files(DIR, labels):
    all = []
    data = []
    for class_id in labels:
        for file in listdir(DIR + class_id):
            fh = open(DIR + class_id + '/' + file, 'r')
            text = fh.read()
            text = text.split()
            all.append(text)
            data.append((text, class_id))
    all = [item for sublist in all for item in sublist]
    vocab = set(all)
    return vocab, data

def vectorize(vocab, sent):
    vector = [0] * len(vocab)
    for word in vocab:
        if word in sent:
            vector[ sent.index(word) ] = 1
    return vector

DIR_TRAIN = "aclImdb/train/"
DIR_TEST = "aclImdb/test/"
labels = ['POS', 'NEG']

vocab, data = read_files(DIR_TRAIN, labels)
vectors = []
labels = []
file = open('train_vectors.txt', 'a')
for el in data:
    file.write(el[1] + '\t' + str(vectorize(vocab, el[0])) + '\n')
# clf = LogisticRegression()
# clf.fit(vectors, labels)

__, test_data = read_files(DIR_TEST, labels)
test_vectors = []
test_labels = []
file = open('test_vectors.txt', 'a')
for el in test_data:
    file.write(el[1] + '\t' + str(vectorize(vocab, el[0])) + '\n')

# print clf.score(test_vectors,test_labels)