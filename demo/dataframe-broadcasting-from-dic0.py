import pandas as pd

scores = [10, 20, 30, 40 ,50]
data = { 'score': scores, 'grade': 'F' }
results = pd.DataFrame(data, columns=['score','grade'])
print(results)

#change column name
results.columns = ['new score', 'new grade']
print(results)

#change index
results.index = ['X1', 'X2','X3','X4','X5']
print(results)
