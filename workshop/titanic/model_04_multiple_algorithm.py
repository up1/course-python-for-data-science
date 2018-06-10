import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def process_with_ticket(data):
    data.drop('Ticket', inplace=True, axis=1)
    return data

def process_with_family(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    #Create feature by size of family (single, small and large)
    data['Single'] = data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    data['Small'] = data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    data['Large'] = data['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

    #Drop useless feature
    data.drop('SibSp', inplace=True, axis=1)
    data.drop('Parch', inplace=True, axis=1)

    return data

def process_with_pclass(data):
    #Encoded to numerical variables with dummy
    pclass_dummies = pd.get_dummies(data['Pclass'], prefix='Pclass')
    data = pd.concat([data, pclass_dummies], axis=1)

    #Drop useless feature
    data.drop('Pclass', inplace=True, axis=1)

    return data

def process_with_cabin(data):
    data['Cabin'] = data['Cabin'].fillna('U')
    data['Cabin'] = data['Cabin'].map(lambda v:v[0])

    #Encoded to numerical variables with dummy
    cabin_dummies = pd.get_dummies(data['Cabin'], prefix='Cabin')
    data = pd.concat([data, cabin_dummies], axis=1)

    #Drop useless feature
    data.drop('Cabin', inplace=True, axis=1)

    return data

def process_with_embarked(data):
    data['Embarked'] = data['Embarked'].fillna('S')

    #Encoded to numerical variables with dummy
    embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = pd.concat([data, embarked_dummies], axis=1)

    #Drop useless feature
    data.drop('Embarked', inplace=True, axis=1)
    return data

def process_with_fare(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    return data

def process_with_age(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    return data

def process_with_sex(data):
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    return data

def get_title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    return ""

def process_with_name(data):
    data['Title'] = data['Name'].apply(get_title)
    data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
    data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                           'Major', 'Lady', 'Sir', 'Col',
                                           'Capt', 'Countess', 'Jonkheer'],'Special')

    #Drop useless feature
    data.drop('Name', inplace=True, axis=1)

    #Encoded to numerical variables with dummy
    title_dummies = pd.get_dummies(data['Title'], prefix='Title')
    data = pd.concat([data, title_dummies], axis=1)

    #Drop useless feature
    data.drop('Title', inplace=True, axis=1)

    return data

def combine_data():
    train_data = pd.read_csv('train.csv')
    testing_data = pd.read_csv('test.csv')
    survived = train_data['Survived']
    train_data.drop('Survived', inplace=True, axis=1)
    data = pd.concat([train_data, testing_data], sort=False)
    data.drop(['PassengerId'], inplace=True, axis=1)
    return data, survived

def compare_with_multiple_algorithms(data, survived):
    training_data = data.iloc[:891].copy()
    testing_data = data.iloc[891:].copy()

    models = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]
    for model in models:
        value = cross_val_score(model, training_data, survived, cv=5, scoring='accuracy')
        print(model.__class__ , " have score ", np.mean(value))

if __name__== "__main__":
    data, survived = combine_data()
    data = process_with_name(data)
    data = process_with_sex(data)
    data = process_with_age(data)
    data = process_with_fare(data)
    data = process_with_embarked(data)
    data = process_with_cabin(data)
    data = process_with_pclass(data)
    data = process_with_family(data)
    data = process_with_ticket(data)
    compare_with_multiple_algorithms(data, survived)
    print('Process is done.')
