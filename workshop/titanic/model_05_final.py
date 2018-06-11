import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from matplotlib import pyplot as plt

def process_with_ticket(data):
    data.drop('Ticket', inplace=True, axis=1)
    return data

def process_with_family(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    #Create feature by size of family
    data['Family'] = data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    dummies = pd.get_dummies(data['Family'], prefix='Family')
    data = pd.concat([data, dummies], axis=1)


    #Drop useless feature
    data.drop('SibSp', inplace=True, axis=1)
    data.drop('Parch', inplace=True, axis=1)
    data.drop('FamilySize', inplace=True, axis=1)
    data.drop('Family', inplace=True, axis=1)

    return data

def process_with_pclass(data):
    #Encoded to numerical variables with dummy
    pclass_dummies = pd.get_dummies(data['Pclass'], prefix='Pclass')
    data = pd.concat([data, pclass_dummies], axis=1)

    #Drop useless feature
    data.drop('Pclass', inplace=True, axis=1)

    return data

def process_with_fare(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    return data

def process_with_age_sex(data):
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Person'] = np.where((data['Age'] <= 16), 0, np.where((data['Age'] > 16) & (data['Sex'] == 0), 1, 2))
    dummies = pd.get_dummies(data['Person'], prefix='Person')
    data = pd.concat([data, dummies], axis=1)

    data.drop('Person', inplace=True, axis=1)
    data.drop('Sex', inplace=True, axis=1)
    data.drop('Age', inplace=True, axis=1)
    return data

def combine_data():
    train_data = pd.read_csv('train.csv')
    testing_data = pd.read_csv('test.csv')
    survived = train_data['Survived']
    train_data.drop('Survived', inplace=True, axis=1)
    data = pd.concat([train_data, testing_data], sort=False)
    data.drop(['PassengerId'], inplace=True, axis=1)
    return data, survived

def create_model(data, survived):
    training_data = data.iloc[:891].copy()
    testing_data = data.iloc[891:].copy()

    model = RandomForestClassifier(n_estimators=100)
    value = cross_val_score(model, training_data, survived, cv=5, scoring='accuracy')
    print(np.mean(value))

    model.fit(training_data, survived)
    prediction_results = model.predict(testing_data)
    return prediction_results

def create_file_for_summission(prediction_results, file_name):
    testing = pd.read_csv('test.csv')
    my_submission = pd.DataFrame({
        'PassengerId': testing['PassengerId'],
        'Survived': prediction_results
    })
    my_submission.to_csv(file_name, index=False)

if __name__== "__main__":
    data, survived = combine_data()
    data.drop('Name', inplace=True, axis=1)
    data.drop('Cabin', inplace=True, axis=1)
    data.drop('Ticket', inplace=True, axis=1)
    data.drop('Embarked', inplace=True, axis=1)
    data.drop('Fare', inplace=True, axis=1)
    data = process_with_age_sex(data)
    data = process_with_pclass(data)
    data = process_with_family(data)
    print(data.info())
    prediction_results = create_model(data, survived)
    create_file_for_summission(prediction_results, "08.csv")
    print('Process is done.')
