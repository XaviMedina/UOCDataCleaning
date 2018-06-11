#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import pandas
import pandas as pd
# Matplotlib
from matplotlib import pyplot as plt
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import scipy.stats
matplotlib.style.use('ggplot')

def explore_dataset(dataset):
    print dataset.describe()

def get_titles(dataset):

    # Obtenim el títol del nom i creem una nova columna
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # Creem un diccionari per mapejar els titols a una categoria social
    dictionary = {
        "Capt": "Oficial",
        "Col": "Oficial",
        "Major": "Oficial",
        "Jonkheer": "Reialesa",
        "Don": "Reialesa",
        "Sir": "Reialesa",
        "Dr": "Oficial",
        "Rev": "Oficial",
        "the Countess": "Reialesa",
        "Dona": "Reialesa",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Reialesa"

    }

    # Mapejem els valors per normalitzar-los
    dataset['Title'] = dataset.Title.map(dictionary)
    return dataset


def show_survive_relation_by_feature(dataset, feature):
    dataset['Died'] = 1 - dataset['Survived']
    dataset.groupby(feature).agg('sum')[['Survived', 'Died']]\
        .plot(kind='bar', figsize=(20, 6), stacked=True, color=['g', 'r'])
    plt.show(block=True)


def show_scatter_plot_by_features(dataset, feature1, feature2):
    plt.figure(figsize=(20, 6))
    ax = plt.subplot()

    ax.scatter(dataset[dataset['Survived'] == 1][feature1], dataset[dataset['Survived'] == 1][feature2],
               c='green', s=dataset[dataset['Survived'] == 1][feature2])
    ax.scatter(dataset[dataset['Survived'] == 0][feature1], dataset[dataset['Survived'] == 0][feature2],
               c='red', s=dataset[dataset['Survived'] == 0][feature2])
    plt.show(block=True)

def anderson_darling_test(dataset):
    anderson_results = scipy.stats.anderson(dataset)
    print(anderson_results)

def fligner_test(dataset1, dataset2):
    fligner_results = scipy.stats.fligner(dataset1, dataset2)
    print(fligner_results)

def show_variable_relation_with_survival(dataset, model):
    features = pd.DataFrame()
    features['feature'] = dataset.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    features.plot(kind='barh', figsize=(10, 10))
    plt.show(block=True)

