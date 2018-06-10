import pandas as pd

#Create dataframe from Dictionary
data = { 'name': ['A', 'B', 'C'],
         'age': [25, 30, 28],
         'salary': [10000, 20000, 25000]
}

df = pd.DataFrame(data)
print(df)

df = pd.DataFrame(data, columns=['name', 'age', 'salary'])
print(df)
