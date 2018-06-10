import pandas as pd

users = pd.read_csv('data.csv', index_col=0)
print(users)
print(users.info())

users['new'] = "F"
print(users)
