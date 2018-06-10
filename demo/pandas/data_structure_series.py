import pandas as pd


#Series
datas = pd.Series(['A', 'B', 'C', 'D'])
datas = pd.Series(['A', 'B', 'C', 'D'], index=[0, 1, 2, 3])

numbers = pd.Series([1, 2, 3, 4])
print(numbers)
print(numbers.values)
print(numbers.index)

# Add index to series
numbers = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(numbers)
print(numbers.values)
print(numbers.index)

# Create series from dictionary
