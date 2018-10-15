#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 21:26:07 2018

@author: pan
"""

from skrvm import RVC
from sklearn.datasets import load_iris

clf = RVC()
iris=load_iris()
clf.fit(iris.data, iris.target)
#RVC(alpha=1e-06, beta=1e-06, beta_fixed=False, bias_used=True, coef0=0.0,
#coef1=None, degree=3, kernel='rbf', n_iter=3000, n_iter_posterior=50,
#threshold_alpha=1000000000.0, tol=0.001, verbose=False)
clf.score(iris.data, iris.target)