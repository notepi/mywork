#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:42:26 2018

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
    rootfile="./data"
    Paths,dirs,AllFile = file_name("./data")
    
    #读跟目录下所有的目录
    dirl=[]
    for paths, dirs, files in os.walk("./data"): 
        break
        pass
    #对目录进行拼接
    cc = [rootfile+'/'+x for x in dirs]
    print("=====")
    #读取每个目录下的文件名，然后叠加
    filel=[]
    for name in cc:
        Paths,dirs,AllFile = file_name(name) 
        filel=filel+[name+'/'+i for i in AllFile if ".DS_S" not in i]
        pass
    #文件名升序排列
    filel=sorted(filel)
    c=[]
    for i in filel:
        a=(int(i.split("/")[-1].split(".")[0].split("_")[-1]))
        if a>= 20150601 and a<= 20180531:
            c.append(i)
            pass
#        break
        pass
    filel=c
    AllData=pd.DataFrame()
    for i in filel:
        print(i)
        temp=pd.read_csv(i)
#        if len(temp)<1:
#            break
#            pass
        temp['date']=temp.date.astype(str)+temp.hour.astype(str)
        AllData=pd.concat([AllData,temp])
#        break
        pass
    #只选取北京
    AllData=AllData.iloc[:,:4]
    a=AllData.columns.values.tolist()
    a[-1]="values"
    AllData.columns=a
#    #最后一个文件
    AllData.to_csv("city.csv",encoding='GBK',index=False)
    
    
    
    
    
    
    