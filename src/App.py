#!/usr/bin/env python
# -*- coding: utf-8 -*-

import DataCleaning
import DataExploration
import DataImport
import Prediction

# Carreguem les dades
test_ds = DataImport.import_test_data()
train_ds = DataImport.import_train_data()

# Explorem la relació de les columnes amb la variable Survived
# DataExploration.show_survive_relation_by_feature(train_ds, 'Sex')
# DataExploration.show_survive_relation_by_feature(train_ds, 'Age')
# DataExploration.show_scatter_plot_by_features(train_ds, 'Fare', 'Age')
combined_ds = DataImport.combine_datasets(train_ds, test_ds)

# Fem una exploració de les dades per a veure si tenim valors nulls
DataExploration.explore_dataset(combined_ds)
combined_ds = DataExploration.get_titles(combined_ds)

# Netejem les dades
dataCleaningObj = DataCleaning.DataCleaning(combined_ds)
combined_ds = dataCleaningObj.clean()

# Test Anderson
DataExploration.anderson_darling_test(combined_ds['Age'])
DataExploration.anderson_darling_test(combined_ds['Fare'])
DataExploration.anderson_darling_test(combined_ds['Sex'])
DataExploration.anderson_darling_test(combined_ds['Family'])

# Test Fligner
DataExploration.fligner_test(combined_ds['Fare'], combined_ds['Pclass'])
combined_ds.drop('Pclass', axis=1, inplace=True)
combined_ds.to_csv('data/cleaned_processed_data.csv', index=False)
# Create and score model
Prediction.score_model(combined_ds)




