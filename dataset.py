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

BOOLEAN = [
    'external',
    'internal',
    'foreign_language',
    'too_short',
    'repeated_characters',
    'repeated_lines',
    'html',
    'phone_number',
    'links',
    'author_valid_email_address',
    # 'author_spam', # empty column
]


NUMERICAL = [
    'author_account_age',
    'author_n_logins',
    'author_uploaded_docs',
    'author_published_docs',
    'document_n_words',
]


def vectorize_features(df, use_cats, encode_cats, scaled):
    bool_X = df[BOOLEAN].values
    num_X = df[NUMERICAL].values
    ncat_X = np.hstack((bool_X, num_X))

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

        X = np.hstack((ncat_X, cat_X))
    else:
        X = ncat_X

    if scaled:
        X = preprocessing.scale(X, with_mean=False)

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
    if encode_cats:
        fname = 'enc_' + fname

    # TODO: remove False
    if os.path.exists(fname):
        ds = pickle.load(open(fname, 'rb'))
    else:    
        df = pd.read_csv('training_dataset.csv')
        X = vectorize_features(df, use_cats, encode_cats, scaled)
        y = df.spam.values
        ds = train_test_split(X, y, test_size=0.2)
        pickle.dump(ds, open(fname, 'wb'))

    return ds

if __name__ == '__main__':
    load_or_create_dataset(scaled=False)
    load_or_create_dataset(scaled=True)

