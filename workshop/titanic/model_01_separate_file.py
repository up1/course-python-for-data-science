import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1 :: Training data
train_data = pd.read_csv('train.csv')
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0

train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2

# Step 2 building predictor with LogisticRegression algirithm
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
algorithm = LogisticRegression(random_state = 1)
algorithm.fit(train_data[predictors], train_data['Survived'])

# Step 3 Testing data
testing_data = pd.read_csv('test.csv')
testing_data['Age'] = testing_data['Age'].fillna(testing_data['Age'].median())
testing_data['Fare'] = testing_data['Fare'].fillna(testing_data['Fare'].median())

testing_data.loc[testing_data['Sex'] == 'female', 'Sex'] = 1
testing_data.loc[testing_data['Sex'] == 'male', 'Sex'] = 0

testing_data['Embarked'] = testing_data['Embarked'].fillna('S')
testing_data.loc[testing_data['Embarked'] == 'S', 'Embarked'] = 0
testing_data.loc[testing_data['Embarked'] == 'C', 'Embarked'] = 1
testing_data.loc[testing_data['Embarked'] == 'Q', 'Embarked'] = 2

print(train_data.head())
# print(testing_data.info())

# Step 4 Try to predict
survived = train_data['Survived'].dropna()
x_train, x_test, y_train, y_test = train_test_split(train_data[predictors], survived, test_size=0.20, random_state=42)
algorithm.fit(x_train, y_train)
score = algorithm.score(x_test, y_test)
print(score)

prediction_results = algorithm.predict(testing_data[predictors])

# Step 5 Create result file for submit
my_submission = pd.DataFrame({
    'PassengerId': testing_data['PassengerId'],
    'Survived': prediction_results
})
my_submission.to_csv('01.csv', index=False)
