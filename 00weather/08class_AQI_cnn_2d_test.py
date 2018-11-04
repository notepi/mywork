#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:43:16 2018

@author: pan
"""

import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix, classification_report,precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# 模型评价 混淆矩阵
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from time import time
from scipy.stats import randint as sp_randint
import math
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint as sp_randint
from keras.models import Sequential
from keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,Flatten,Conv1D,MaxPooling1D,Activation
#from keras.datasets import mnist 
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio 
from keras.wrappers.scikit_learn import KerasClassifier

def file_name(file_dir):  
  for paths, dirs, files in os.walk(file_dir): 
#    print(paths) #当前目录路径 
#    print(dirs) #当前路径下所有子目录 
#    print(files) #当前路径下所有非目录子文件 
    pass
  return paths,dirs,files  
# Utility function to report best scores
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
            pass
        pass
    pass

def make_model(dense_layer_sizes,cnn_layer, Neuron, filters, kernel_size, pool_size):
    '''Creates model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes.
        This list has one number for each layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling
    '''
#    kernel_size=9
    
    print("dense_layer_sizes",dense_layer_sizes)
    print("Neuron",Neuron)
    print("filters",filters)
    print("kernel_size",kernel_size)
    print("pool_size",pool_size)
    
    model = Sequential()
    model.add(Conv2D(filters, kernel_size,
                     padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    for i in range(cnn_layer):
        #层数越深,filters越多
        model.add(Conv2D(filters*(i+1), kernel_size,
                         padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        pass


    model.add(Flatten())
    for layer_size in range(dense_layer_sizes):
        model.add(Dense(Neuron))
        model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    #打印出模型概况，它实际调用的是keras.utils.print_summary
    model.summary()

    return model

if __name__ == "__main__":
    file="./mergefinal/AQI/AQI.csv"
    Data=pd.read_csv(file,encoding = "GBK")
    del Data['date']
    
    tag=[]
    for i in Data["values"]:
        if i<50:
            tag.append(0)
            pass
        elif i <101:
            tag.append(1)
            pass
        elif i <151:
            tag.append(2)
            pass
        elif i <201:
            tag.append(3)
            pass
        elif i <300:
            tag.append(4)
            pass
        else:
            tag.append(5)
            pass
        pass
    Data['tag']=tag
    del Data["values"]
    

    
#做样本不均衡处理
#    #生成训练数据和测试数据 
    cc=[]
    a=Data.tag.value_counts().index.tolist() 
    
    #根据种类进行分类
    DataTest = pd.DataFrame()
    DataTrain = pd.DataFrame()
    for i in a:
        temp=Data[Data["tag"]==i]

        TemptestlDataTest, TempTrainlDataTrain = train_test_split(temp, train_size=0.2, random_state=1)
        DataTest=pd.concat([DataTest,TemptestlDataTest])
        DataTrain=pd.concat([DataTrain,TempTrainlDataTrain])
        
        pass
    
    ###    #生成训练数据和测试数据
##    testlDataTest, TrainlDataTrain = train_test_split(AllData, train_size=0.3, random_state=1)
     
    XdataTrain = DataTrain.iloc[:,:-1]
    TagTrain = DataTrain.iloc[:,-1]
    
    XdataTest = DataTest.iloc[:,:-1]
    TagTest = DataTest.iloc[:,-1]
    
    
    #拓展维度
    XdataTrain['674']=0
    XdataTest['674']=0
    XdataTrain['675']=0
    XdataTest['675']=0
    XdataTrain['676']=0
    XdataTest['676']=0
    
    XdataTrain=XdataTrain.values.reshape(XdataTrain.shape[0],26,26)
    XdataTest=XdataTest.values.reshape(XdataTest.shape[0],26,26)
    
    XdataTrain=np.expand_dims(XdataTrain, axis=len(XdataTrain.shape))
    XdataTest=np.expand_dims(XdataTest, axis=len(XdataTest.shape))
    
    input_shape=tuple(XdataTest.shape[1:])
    
    #cnn
    print("============================CNN=========================================")
    enc = OneHotEncoder()
    enc.fit(TagTrain.values.reshape(-1,1))
    TagTest = enc.transform(TagTest.values.reshape(-1,1)).toarray()
    TagTrain = enc.transform(TagTrain.values.reshape(-1,1)).toarray()
    
    num_classes=TagTest.shape[1]
    
    
    
    dense_size_candidates = np.arange(2,2*10,2)
    cnn_layers = np.arange(1,1*5,1)
    epochs_size = np.arange(3,3*8,3)
    filters_size = np.arange(16,16*5,16)
    Neuron_size=np.arange(32,32*5,32)
    kernel_size= [3,5,7,9]
    
    my_classifier = KerasClassifier(make_model, batch_size=32)
    validator = RandomizedSearchCV(my_classifier,
                             param_distributions={'dense_layer_sizes': dense_size_candidates,
                                         # epochs is avail for tuning even when not
                                         # an argument to model building function
                                         'epochs': epochs_size,
                                         'filters': filters_size,
                                         'Neuron': Neuron_size,
                                         'cnn_layer':cnn_layers,
                                         'kernel_size': kernel_size,
                                         'pool_size': [2]},
                             scoring='neg_log_loss',
                             n_jobs=1)
    
    validator.fit(XdataTrain, TagTrain)
    
#    print('The parameters of the best model are: ')
#    print(validator.best_params_)
#    
#    # validator.best_estimator_ returns sklearn-wrapped version of best model.
#    # validator.best_estimator_.model returns the (unwrapped) keras model
#    best_model = validator.best_estimator_.model
#    metric_names = best_model.metrics_names
#    metric_values = best_model.evaluate(XdataTest, TagTest)
#    for metric, value in zip(metric_names, metric_values):
#        print(metric, ': ', value)
    
    
    
    
    
    
    
#    ##############
#    # 训练模型，以 32 个样本为一个 batch 进行迭代
#    model.fit(XdataTrain, TagTrain, epochs=100, batch_size=32)
#
#    #测试
#    #测试结果
#    print("========")
#    loss_and_metrics = model.evaluate(XdataTest, TagTest, batch_size=10)
#    print("test result is:", loss_and_metrics[1])    
    
################################################################################

