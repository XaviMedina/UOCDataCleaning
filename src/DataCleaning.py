#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import Numpy
import numpy as np
# Import pandas
import pandas as pd

# Eliminem els warnings de pandas
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


class DataCleaning:

    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset

    def clean(self):
        self.clean_age()
        self.clean_names()
        self.clean_fares()
        self.clean_embarked()
        self.clean_cabin()
        self.clean_sex()
        self.clean_pclass()
        self.clean_family()
        self.clean_ticket()
        self.dataset.drop('PassengerId', inplace=True, axis=1)
        return self.dataset

    def clean_age(self):
        group_by_columns = ['Sex', 'Pclass', 'Title']
        # Agrupem les dades per sexe, classe i títol
        grouped_train = self.dataset.head(891).groupby(group_by_columns)
        grouped_median_train = grouped_train.median()
        grouped_test = self.dataset.iloc[891:].groupby(group_by_columns)
        grouped_median_test = grouped_test.median()

        self.dataset.head(891).Age = self.dataset.head(891).apply(
            lambda r: self.fill_age_by_sex_and_class(r, grouped_median_train) if np.isnan(r['Age'])
            else r['Age'], axis=1)

        self.dataset.iloc[891:].Age = self.dataset.iloc[891:].apply(
            lambda r: self.fill_age_by_sex_and_class(r, grouped_median_test) if np.isnan(r['Age'])
            else r['Age'], axis=1)

    def fill_age_by_sex_and_class(self, inspected_row, group_median):
        if inspected_row['Sex'] == 'female' and inspected_row['Pclass'] == 1:
            if inspected_row['Title'] == 'Miss':
                return group_median.loc['female', 1, 'Miss']['Age']
            elif inspected_row['Title'] == 'Mrs':
                return group_median.loc['female', 1, 'Mrs']['Age']
            elif inspected_row['Title'] == 'Oficial':
                return group_median.loc['female', 1, 'Oficial']['Age']
            elif inspected_row['Title'] == 'Reialesa':
                return group_median.loc['female', 1, 'Reialesa']['Age']
        elif inspected_row['Sex'] == 'female' and inspected_row['Pclass'] == 2:
            if inspected_row['Title'] == 'Miss':
                return group_median.loc['female', 2, 'Miss']['Age']
            elif inspected_row['Title'] == 'Mrs':
                return group_median.loc['female', 2, 'Mrs']['Age']
        elif inspected_row['Sex'] == 'female' and inspected_row['Pclass'] == 3:
            if inspected_row['Title'] == 'Miss':
                return group_median.loc['female', 3, 'Miss']['Age']
            elif inspected_row['Title'] == 'Mrs':
                return group_median.loc['female', 3, 'Mrs']['Age']
        elif inspected_row['Sex'] == 'male' and inspected_row['Pclass'] == 1:
            if inspected_row['Title'] == 'Master':
                return group_median.loc['male', 1, 'Master']['Age']
            elif inspected_row['Title'] == 'Mr':
                return group_median.loc['male', 1, 'Mr']['Age']
            elif inspected_row['Title'] == 'Oficial':
                return group_median.loc['male', 1, 'Oficial']['Age']
            elif inspected_row['Title'] == 'Reialesa':
                return group_median.loc['male', 1, 'Reialesa']['Age']
        elif inspected_row['Sex'] == 'male' and inspected_row['Pclass'] == 2:
            if inspected_row['Title'] == 'Master':
                return group_median.loc['male', 2, 'Master']['Age']
            elif inspected_row['Title'] == 'Mr':
                return group_median.loc['male', 2, 'Mr']['Age']
            elif inspected_row['Title'] == 'Oficial':
                return group_median.loc['male', 2, 'Oficial']['Age']
        elif inspected_row['Sex'] == 'male' and inspected_row['Pclass'] == 3:
            if inspected_row['Title'] == 'Master':
                return group_median.loc['male', 3, 'Master']['Age']
            elif inspected_row['Title'] == 'Mr':
                return group_median.loc['male', 3, 'Mr']['Age']

    def clean_names(self):
        # Eliminem la columna Name
        self.dataset.drop('Name', axis=1, inplace=True)
        # Creem una variable dummy dels títols
        titles_dummies = pd.get_dummies(self.dataset['Title'], prefix='Title')
        # Combinem la nova columna amb el dataset
        self.dataset = pd.concat([self.dataset, titles_dummies], axis=1)
        # Eliminem l'antiga columna del títol
        self.dataset.drop('Title', axis=1, inplace=True)

    def clean_fares(self):
        self.dataset.head(891).Fare.fillna(self.dataset.head(891).Fare.mean(), inplace=True)
        self.dataset.iloc[891:].Fare.fillna(self.dataset.iloc[891:].Fare.mean(), inplace=True)

    def clean_embarked(self):
        # Omplim els valors buits amb el valor més típic
        self.dataset.head(891).Embarked.fillna('S', inplace=True)
        self.dataset.iloc[891:].Embarked.fillna('S', inplace=True)
        # Creem les variables dummy per a normalitzar els valors
        embarked_dummies = pd.get_dummies(self.dataset['Embarked'], prefix='Embarked')
        self.dataset = pd.concat([self.dataset, embarked_dummies], axis=1)
        # Eliminem la columna de Embarked
        self.dataset.drop('Embarked', axis=1, inplace=True)

    def clean_cabin(self):
        # Omplim els valors buits amb Unknown
        self.dataset.Cabin.fillna('U', inplace=True)
        # Modifiquem els valors de la cabina per a quedar-nos amb la lletra
        self.dataset['Cabin'] = self.dataset['Cabin'].map(lambda c: c[0])

        # Fem un dummy encoding de la variable
        cabin_dummies = pd.get_dummies(self.dataset['Cabin'], prefix='Cabin')
        self.dataset = pd.concat([self.dataset, cabin_dummies], axis=1)
        # Eliminem la columna original
        self.dataset.drop('Cabin', axis=1, inplace=True)

    def clean_sex(self):
        # Normalitzem els valors a númeric per a que sigui més fàcil treballar
        self.dataset['Sex'] = self.dataset['Sex'].map({'male': 1, 'female': 0})

    def clean_pclass(self):
        # Creem les variables dummy per Pclass
        pclass_dummies = pd.get_dummies(self.dataset['Pclass'], prefix="Pclass")
        self.dataset = pd.concat([self.dataset, pclass_dummies], axis=1)

    def extract_ticket_number(self, ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = filter(lambda t: not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'U'

    def clean_ticket(self):
        # Extraiem els prefixos dels tiquets, si no en té, possem 'U' per Unknwon
        self.dataset['Ticket'] = self.dataset['Ticket'].map(self.extract_ticket_number)
        # Creem les variables dummy per a les numeracions dels tiquets
        tickets_dummies = pd.get_dummies(self.dataset['Ticket'], prefix='Ticket')
        self.dataset = pd.concat([self.dataset, tickets_dummies], axis=1)
        # Eliminem la columna Ticket
        self.dataset.drop('Ticket', inplace=True, axis=1)

    def clean_family(self):
        # Creem una nova columna, que combina el nombre de germans + esposes + fills, així tenim una variable amb
        #  el tamany de la familia
        self.dataset['Family'] = self.dataset['Parch'] + self.dataset['SibSp'] + 1




