import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
data = pd.read_table('spam.csv', encoding='latin-1', header=None)
df = pd.DataFrame(data=(s.split(',', 1) for s in data[0][1:]), columns=['class','text'])
df['class_id'] = df['class']=='spam'

data = []
ALL = []
for i in xrange(0, len(df)):
    text = df.text[i].split()
    label = df.class_id[i]
    data.append((text, label))
    ALL += text

vocab = list(set(ALL))

vectors = []
labels = []

for el in data:
    vector = [0] * len(vocab)
    for word in vocab:
        if word in el[0]:
            vector[vocab.index(word)] = 1
    vectors.append(vector)
    labels.append(el[1])

x_train, y_train = vectors[: int(0.8 * len(vectors)) ], labels [: int(0.8 * len(labels)) ]
x_test, y_check = vectors[int(0.8 * len(vectors)) :], labels [ int(0.8 * len(labels)) :]

# clf = BernoulliNB()
clf = SVC()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
correct = 0
for i in xrange(0, len(y_check)):
    if y_check[i] == pred[i]:
        correct += 1