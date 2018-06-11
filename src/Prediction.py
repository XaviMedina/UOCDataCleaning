#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
# Import Numpy
import numpy as np
# Import pandas
import pandas as pd
# Import de sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

import DataExploration

matplotlib.style.use('ggplot')

def score_model(dataset):
    train, test, targets = recover_train_test_target(dataset)

    randomForestClassifier = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    randomForestClassifier = randomForestClassifier.fit(train, targets)

    DataExploration.show_variable_relation_with_survival(train, randomForestClassifier)

    model = SelectFromModel(randomForestClassifier, prefit=True)
    train_reduced = model.transform(train)
    test_reduced = model.transform(test)

    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)

    print 'Score: ', compute_score(model, train, targets, scoring='accuracy')

    output = model.predict(test).astype(int)
    df_output = pd.DataFrame()
    aux = pd.read_csv('data/test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId', 'Survived']].to_csv('data/solution.csv', index=False)


def recover_train_test_target(dataset):

    train_original = pd.read_csv('data/train.csv')

    targets = train_original.Survived
    train = dataset.head(891)
    test = dataset.iloc[891:]

    return train, test, targets

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)