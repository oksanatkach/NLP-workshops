import itertools

POS_sents = open('aclImdb/train-pos.txt', 'r').read().split('\n')
POS_words = list(itertools.chain([[sent.split()] for sent in POS_sents ]))
print POS_words