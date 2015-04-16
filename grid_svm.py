"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.grid_search.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from dataset import load_or_create_dataset
print(__doc__)
import numpy as np

X, y = load_or_create_dataset()


# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Set the parameters by cross-validation
tuned_parameters = [
                    {'kernel': ['linear'], 'gamma': [50, 100, 500], 'C': [0.1, 1, 10]},
                    {'kernel': ['rbf'], 'gamma': [100, 200, 300, 500], 'C': [0.1, 1, 10]}
                    # {'kernel': ['poly'], 'gamma': [1, 10], 'C': [1, 10]},
                    # {'kernel': ['rbf'], 'gamma': [0.1, 1, 0.01], 'C': [1, 10]},
                    # {'kernel': ['rbf'], 'gamma': np.arange(0.00001, 0.0001, 0.00002), 'C': [0.01, 0.03, 0.05, 0.1]}
                ]
 
# {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
# {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},

scores = ['precision']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3,
                       scoring=score, n_jobs=-1)


    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.