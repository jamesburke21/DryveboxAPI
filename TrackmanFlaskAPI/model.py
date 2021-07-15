# ML imports
import json
import math
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from PIL import Image
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from util import plot_roc


class cpModel(object):

    def __init__(self):
        """Decision Tree Classifier
        Attributes:
            clf: sklearn classifier model
        """
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        """Fits a TFIDF vectorizer to the text
        """
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """Trains the classifier to associate the label with the sparse matrix
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Predict class probabilities for X.
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1] #check output

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='chalicelib/models/TFIDFVectorizer.pkl'):
        """Saves the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='TrackmanAPI/models/cP.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))

    def plot_roc(self, X, y, size_x, size_y):
        """Plot the ROC curve for X_test and y_test.
        """
        plot_roc(self.clf, X, y, size_x, size_y)


# class stModel(object):
#
#     def __init__(self):
#         """Decision Tree Classifier
#         Attributes:
#             clf: sklearn classifier model
#         """
#         self.clf = MultinomialNB()
#
#     def train(self, X, y):
#         """Trains the classifier to associate the label with the sparse matrix
#         """
#         X_train, X_test, y_train, y_test = train_test_split(X, y)
#         self.clf.fit(X, y)
#
#     def predict_proba(self, X):
#         """Returns probability for the binary class '1' in a numpy array
#         """
#         y_proba = self.clf.predict_proba(X)
#         return y_proba[:, 1]
#
#     def predict(self, X):
#         """Returns the predicted class in an array
#         """
#         y_pred = self.clf.predict(X)
#         return y_pred
#
#     def pickle_clf(self, path='TrackmanAPI/models/cP.pkl'):
#         """Saves the trained classifier for future use.
#         """
#         with open(path, 'wb') as f:
#             pickle.dump(self.clf, f)
#             print("Pickled classifier at {}".format(path))
#
#     def plot_roc(self, X, y, size_x, size_y):
#         """Plot the ROC curve for X_test and y_test.
#         """
#         plot_roc(self.clf, X, y, size_x, size_y)
