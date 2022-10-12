import pandas as pd
import numpy as np
data = pd.read_table('spam.csv', encoding='latin-1', header=None)
df = pd.DataFrame(data=(s.split(',', 1) for s in data[0][1:]), columns=['class','text'])
df['class_id'] = df['class']=='spam'

s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])