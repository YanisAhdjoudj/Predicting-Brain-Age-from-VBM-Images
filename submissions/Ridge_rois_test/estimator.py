# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:50:51 2021

@author: yanis
"""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet as en
from sklearn.preprocessing import MinMaxScaler


class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, :284]



class VBMFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, 284:]



def get_estimator():
    """Build your estimator here."""



    eln = en(alpha=1, l1_ratio=0.1, fit_intercept=True,
                   normalize=False, precompute=False, max_iter=1000,
                   copy_X=True, tol=0.0001, warm_start=False, positive=False,
                   random_state=123, selection='cyclic')

    estimator = make_pipeline(VBMFeatureExtractor(), StandardScaler(), eln)

    return estimator