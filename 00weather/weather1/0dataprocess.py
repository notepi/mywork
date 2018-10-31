#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:17:09 2018

@author: pan
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os
def file_name(file_dir):  
  for paths, dirs, files in os.walk(file_dir): 
#    print(paths) #当前目录路径 
#    print(dirs) #当前路径下所有子目录 
#    print(files) #当前路径下所有非目录子文件 
    pass
  return paths,dirs,files  

if __name__ == "__main__":

    rootfile="./data/"
    Paths,dirs,AllFile = file_name(rootfile)
    
    AllFile=[rootfile+i for i in AllFile]
    data=pd.DataFrame()
    
    for i in AllFile:
        data = pd.concat([data,pd.read_excel('./data/20150513-20151231.xls',encoding = "GBK")])
    pass

    