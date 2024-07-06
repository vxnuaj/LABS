import pandas as pd
import numpy as np
import seaborn as sns

'''

CSV FILES

'''

data = pd.read_csv('data.csv') # Reading a CSV per default
data2 = pd.read_csv('data.tsv', sep='\t') # Reading a tsv file with seperators as tabs (\t)
data3 = pd.read_csv('data.csv', header=None) # Don't use first row as headers, ahve none
data4 = pd.read_csv('data.csv', na_values='Single') # allows us to set null values -> NaN
data5 = pd.read_csv('data.csv', names=['age'], header = 0) # sequence of column labels to apply, allowing us to override the current set of headers if needbe
data6 = pd.read_csv('data.csv', usecols=['age']) # allowing us to select which specific column(s) to read. 

'''

JSON FILES

'''

data = pd.read_json('data.json') # reading from a json file into a df
data6.to_json('2json.json', orient='records') # writing a dataframe to a json file
