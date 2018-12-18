#!/usr/bin/python3
# -*- coding utf-8 -*-
"""
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn import svm
from time import time



def classFir(i):
    if i <= 1000:
        return 0
        pass
    elif i > 1000 and i <= 1500:
        return 1
        pass
    elif i > 1500 and i <= 2000:
        return 2
        pass
    elif i > 2000 and i <= 2500:
        return 3
        pass
    elif i > 2500 and i <= 3000:
        return 4
        pass
    elif i > 3000 and i <= 3500:
        return 5
        pass
    elif i > 3500 and i <= 4000:
        return 6
        pass
    elif i > 4000 and i <= 4500:
        return 7
        pass
    elif i > 4500 and i <= 5000:
        return 8
        pass
    elif i > 5000:
        return 9
        pass
    else:
        print(i)
    pass

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":
    AllData = pd.read_csv('./finaldata0911.csv')
    # 对预测值进行编码
    AllData.counts = AllData.counts.apply(classFir)
    print("=================data process")

    # 对数据进行标准化
    log_AllData = pd.DataFrame()
    name = AllData.columns.tolist()
    for i in name[:-2]:
        log_AllData[i] = preprocessing.scale(list(AllData[i]))
        # break
        pass
    ##############################################################################
    # # %%
    # # logistic回归
    # # 训练模型
    # alpha_can = np.logspace(-3, 2, 10)
    # # 使用SVM作为分类器
    # clf = LogisticRegression(penalty="l1", solver="liblinear")
    # # use a full grid over all parameters
    # param_grid = {"C": np.logspace(-3, 2, 10)}
    # # 网格搜索， grid search
    # grid_search = GridSearchCV(clf, param_grid=param_grid)
    # grid_search.fit(log_AllData, AllData.counts.values.ravel())
    # a = grid_search.best_estimator_.coef_
    #
    # # %%
    # lr = LogisticRegression(penalty='l2', solver="liblinear", multi_class='ovr')
    # lr.fit(log_AllData, AllData.counts.values.ravel())
    # b = lr.coef_

    # # %%
    # # svm训练
    # # 使用SVM作为分类器
    # clf = svm.SVC(kernel="linear")
    #
    # # use a full grid over all parameters
    # param_grid = {"kernel": ["linear", "rbf", "sigmoid"],
    #               "C": np.logspace(-2, 2, 10),
    #               "gamma": np.logspace(-2, 2, 10)
    #               }
    #
    # # 网格搜索， grid search
    # grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3)
    # start = time()
    # grid_search.fit(log_AllData,  AllData.counts.values.ravel())
    #
    # print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
    #       % (time() - start, len(grid_search.cv_results_['params'])))
    # report(grid_search.cv_results_)
    # svc = grid_search

    # %%
    # 随机森林
    # 使用随机森林作为分类器，分类器有20课树
    clf = RandomForestClassifier(n_estimators=20)
    # 设置可能学习的参数
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    # 随机搜索， randomized search
    n_iter_search = 35
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)
    random_search.fit(AllData.iloc[:, :-2], AllData.iloc[:, -1])
    feature_im = random_search.best_estimator_.feature_importances_
    feature_im = pd.DataFrame([name[:-2], list(feature_im)]).T
    feature_im.columns = ["feature", "importance"]
    feature_im = feature_im.sort_values(by="importance")
    feature_im.to_csv("important.csv", header=False, index=False)
    pass
