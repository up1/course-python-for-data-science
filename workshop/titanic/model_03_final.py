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

def try_to_tuning(data, survived):
    training_data = data.iloc[:891].copy()
    testing_data = data.iloc[891:].copy()

    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(training_data, survived)

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

def selection_features(data, survived):
    training_data = data.iloc[:891].copy()
    testing_data = data.iloc[891:].copy()
    model = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    model = model.fit(training_data, survived)

    features = pd.DataFrame()
    features['feature'] = training_data.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    matplotlib.use('Agg')
    fig = features.plot(kind='barh', figsize=(25, 25)).get_figure()
    fig.savefig('test.png')


def create_model(data, survived):
    training_data = data.iloc[:891].copy()
    testing_data = data.iloc[891:].copy()

    parameters = {'bootstrap': False, 'max_depth': 8, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 10}
    model = SVC(kernel="linear", C=0.025)

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
    data = process_with_name(data)
    data = process_with_sex(data)
    data = process_with_age(data)
    data = process_with_fare(data)
    data = process_with_embarked(data)
    data = process_with_cabin(data)
    data = process_with_pclass(data)
    data = process_with_family(data)
    data = process_with_ticket(data)
    selection_features(data, survived)
    # try_to_tuning(data, survived)
    # prediction_results = create_model(data, survived)
    # create_file_for_summission(prediction_results, "06.csv")
    print('Process is done.')
