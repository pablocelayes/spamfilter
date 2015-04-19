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


NUMERICAL = [
    'external',
    'internal',
    'foreign_language',
    'too_short',
    # 'repeated_characters',
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


def vectorize_features(df, use_cats, encode_cats):
    num_X = df[NUMERICAL].values

    if use_cats:
        # Encode categorical features
        catdf = df[CATEGORICAL].copy()
        for f in CATEGORICAL:
            uniquevals = list(catdf[f].unique())
            catdf[f] = catdf[f].map(lambda v: uniquevals.index(v))
        cat_X = catdf.values

        if encode_cats:
            enc = preprocessing.OneHotEncoder()
            cat_X = enc.fit_transform(cat_X)
            cat_X = cat_X.todense()

        # Apply scaling
        X = np.hstack((num_X, cat_X))
    else:
        X = np.hstack((num_X,))
        # X = np.array(num_X)

    return X


def load_or_create_dataset(scaled=True, use_cats=True, encode_cats=True):
    """
        use_cats: whether or not to include categorical features
            (can be useful for testin some probabilistic approaches for
                which we haven't yet implementd proper handling of
                categorical features)
        encode_cats: whether or not to encode categorical features
            as multiple binary features (not necessary for tree-based
                classifiers)
    """
    fname = 'dataset.pickle'
    if scaled:
        fname = 'scaled_' + fname
    if use_cats:
        fname = 'cat_' + fname
    if os.path.exists(fname):
        X, y = pickle.load(open(fname, 'rb'))
    else:    
        df = pd.read_csv('training_dataset.csv')
        X = vectorize_features(df, use_cats, encode_cats)
        if scaled:
            X = preprocessing.scale(X)
        y = df.spam.values
        # pickle.dump((X, y), open(fname, 'wb'))

    return X, y

if __name__ == '__main__':
    load_or_create_dataset(scaled=False)
    load_or_create_dataset(scaled=True)

