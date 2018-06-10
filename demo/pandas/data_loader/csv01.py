import pandas as pd

#Read data from CSV file
data = pd.read_csv('sample01.csv')
print(data)

#Change separater
data = pd.read_csv('sample01.csv', sep=',')
print(data)

#Remove header
data = pd.read_csv('sample01.csv', header=None)
print(data)

#Change name of column
names = ['a', 'b', 'c', 'd', 'message']
data = pd.read_csv('sample01.csv', names=names)
print(data)

#Change index column
data = pd.read_csv('sample01.csv', index_col='message')
print(data)
