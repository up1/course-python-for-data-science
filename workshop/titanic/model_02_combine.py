import pandas as pd
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Step 01 read data and combine
train_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')

survived_of_training = train_data.Survived

data = pd.concat([train_data.drop('Survived', axis=1), testing_data], sort=False)

#  Step 2 :: Missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Step 3 :: Encoding data to number (Numeric data)
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# Step 4 :: Select some columns from data to build model
predictors = ['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']
data = data[predictors]

# Step 5 :: Building Machine Learning Model
data_train = data.iloc[:891]
data_test = data.iloc[891:]

X = data_train.values
test = data_test.values
y = survived_of_training.values

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

prediction_results = clf.predict(test)

# Step 6 Create result file for submit
my_submission = pd.DataFrame({
    'PassengerId': testing_data['PassengerId'],
    'Survived': prediction_results
})
my_submission.to_csv('02.csv', index=False)
