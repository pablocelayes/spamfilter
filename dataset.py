#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
import os
import pickle
from sklearn import preprocessing
import numpy as np


CATEGORICAL = [
    'author_email_domain',
    'author_plan_at_submission',
    'document_format',
    'document_upload_method',
]


NUMERIC_FEATURES = [
    'external',
    'internal',
    'foreign_language',
    'too_short',
    'repeated_characters',
    'repeated_lines',
    'html',
    'phone_number',
    'links',
    'author_account_age',
    'author_n_logins',
    'author_uploaded_docs',
    # 'author_spam', # empty column
    'author_published_docs',
    'author_valid_email_address',
    'document_n_words',
]


def vectorize_features(df):
    num_X = df[NUMERIC_FEATURES].values

    # Encode categorical features
    catdf = df[CATEGORICAL]
    for f in CATEGORICAL:
        uniquevals = list(catdf[f].unique())
        catdf[f] = catdf[f].map(lambda v: uniquevals.index(v))
    
    enc = preprocessing.OneHotEncoder()
    cat_X = enc.fit_transform(catdf.values)
    cat_X = cat_X.todense()

    # Apply scaling
    
    X = np.hstack((num_X, cat_X))

    return X 


def load_or_create_dataset():
    fname = 'dataset.pickle'
    if os.path.exists(fname):
        X_train, X_test, y_train, y_test = pickle.load(open(fname, 'rb'))
    else:    
        df = pd.read_csv('training_dataset.csv')
        X = vectorize_features(df)
        y = df.spam.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        pickle.dump((X_train, X_test, y_train, y_test), open(fname, 'wb'))

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    load_or_create_dataset()

