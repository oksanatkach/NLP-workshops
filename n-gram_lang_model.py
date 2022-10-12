from nltk.util import ngrams
from collections import Counter
from math import log

#

file = 'linux_input.txt'

def ngram_lm(file):

    fh = open(file, 'r')
    text = fh.read()

    unigrams = Counter(ngrams(text,1))
    bigrams = Counter(ngrams(text,2))
    trigrams = Counter(ngrams(text,3))
    fourgrams = Counter(ngrams(text,4))
    fivegrams = Counter(ngrams(text,5))

    uni_mles = { char: log ( unigrams[char] / float(len(unigrams)) ) for char in unigrams }
    bi_mles = { char: log ( bigrams[char] / float(unigrams[char[:-1]]) ) for char in bigrams }
    tri_mles = { char: log ( trigrams[char] / float(bigrams[char[:-1]]) ) for char in trigrams }
    four_mles = { char: log ( fourgrams[char] / float(trigrams[char[:-1]]) ) for char in fourgrams }
    five_mles = { char: log ( fivegrams[char] / float(fourgrams[char[:-1]]) ) for char in fivegrams }
    print uni_mles
    print bi_mles
    print tri_mles
    print four_mles
    print five_mles