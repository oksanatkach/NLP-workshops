import codecs
fname = 'poems.txt'
fh = codecs.open(fname, 'r', 'utf-8')
data = fh.read()
SEP = '♦'