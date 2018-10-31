#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 00:46:48 2018

@author: pan
"""

import numpy as np
import pandas as pd
import os 
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.externals import joblib  

def file_name(file_dir):  
  for paths, dirs, files in os.walk(file_dir): 
#    print(paths) #当前目录路径 
#    print(dirs) #当前路径下所有子目录 
#    print(files) #当前路径下所有非目录子文件 
    pass
  return paths,dirs,files  

if __name__ == "__main__":
    temp=pd.read_csv("./errorf.csv")
    temp=temp.iloc[:,1]
    data=pd.DataFrame()
    errorf=[]
    errord=[]
    for i in temp:
        try:
            data = pd.concat([data,pd.read_csv(i)])
            pass
        except UnicodeDecodeError:
            errorf.append(i)
        except :
            errord.append(i)
#        print(i)
#        break
        pass