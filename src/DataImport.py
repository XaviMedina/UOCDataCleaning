#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import pandas
import pandas as pd


def import_test_data():
    return pd.read_csv('data/test.csv')


def import_train_data():
    return pd.read_csv('data/train.csv')

# Combinar datasets
def combine_datasets(train, test):
    # Eliminem la variable survived del dataset d'entrenament
    train.drop('Survived', 1, inplace=True)

    # Combinem els datasets
    combined_ds = train.append(test)
    combined_ds.reset_index(inplace=True)
    combined_ds.drop('index', inplace=True, axis=1)

    return combined_ds