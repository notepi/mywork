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
from keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,Flatten,Conv1D,MaxPooling1D
#from keras.datasets import mnist 
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio 

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
    
    #画图
#    n, bins, patches = plt.hist(Data['tag'], 11, density=True, facecolor='g', alpha=0.75)
#    
##    plt.figure()
#    plt.xlabel('Times')
#    plt.ylabel('Counts')
#    plt.title('Frequency distribution')
#    plt.grid(True)
#    plt.show()  
    
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
    
    
    
    XdataTrain=np.expand_dims(XdataTrain, axis=2)
    XdataTest=np.expand_dims(XdataTest, axis=2)
    
    #cnn
    print("============================CNN=========================================")
    enc = OneHotEncoder()
    enc.fit(TagTrain.values.reshape(-1,1))
    TagTest = enc.transform(TagTest.values.reshape(-1,1)).toarray()
    TagTrain = enc.transform(TagTrain.values.reshape(-1,1)).toarray()

         
    #model
    #创建模型序列
    model = Sequential()
    #######卷积层
    #25表示卷积核的个数,(3,3)表示卷积核的大小,
    #(28,28,1)表示图片是28x28,一维
##    Conv1D(filters, 
##           kernel_size, 
##           strides=1, 
##           padding='valid', 
##           dilation_rate=1, 
##           activation=None, 
##           use_bias=True, 
##           kernel_initializer='glorot_uniform', 
##           bias_initializer='zeros', 
##           kernel_regularizer=None, 
##           bias_regularizer=None, 
##           activity_regularizer=None, 
##           kernel_constraint=None, 
##           bias_constraint=None)
##    model.add(Conv1D(25,8,input_shape=(None, 38)))  
    model.add(Conv1D(filters=25,kernel_size=8,input_shape=(XdataTrain.shape[1], 1))) 
    #变成25x26x26
    #新的图是26x26,因为边边角消失
    
    #pooling层
    #在(2,2)中取max
    model.add(MaxPooling1D(4))
    #变成25x13x13
    #新的图是13x13,因为（2,2)是在两行两列中取一个出来
    
    #每次的都可以是不一样的
    ######卷积层
    #根据文献，在靠近input部分少，越往后越多
    model.add(Conv1D(51,4)) 
    #见文章http://www.360doc.com/content/18/0305/05/5315_734350634.shtml
    #在卷积核中，并不是每个Featurn map都是用一样的卷积核
    #最后的结果变成50x11x11
    #这是因为在上一次的卷积中是将输入1个通道，变成输出25个通道
    #在第二次时输入就变成了25个通道,设置输出多少个通道都可以
    
    model.add(MaxPooling1D(4))
    #50x5x5
    
    
    #####关于参数的多少
    #第一次通道是1，每个通道中filter是3x3，所以是9,一共是25个filter
    #第二次通道是25，每个通道中filter是3x3，所以是3x3x25，这样是一个filter的参数
    #每个filter把一个多层的通道压成单通道的图
    #新的多通道是由多个filter构成的
    #所以也就可以知道，为何可以随便设置输出了，这样是51,也就是51个filter，每个的
    #参数是3x3x25
    
    #fltten
    model.add(Flatten())
    
    #DNN
    for i in range(2):
        model.add(Dense(100,activation='relu'))
        pass
    
    model.add(Dense(TagTrain.shape[1],activation='softmax'))
    
    # 多分类问题
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    ##############
    # 训练模型，以 32 个样本为一个 batch 进行迭代
    model.fit(XdataTrain, TagTrain, epochs=100, batch_size=32)

    #测试
    #测试结果
    print("========")
    loss_and_metrics = model.evaluate(XdataTest, TagTest, batch_size=10)
    print("test result is:", loss_and_metrics[1])    
    
################################################################################

