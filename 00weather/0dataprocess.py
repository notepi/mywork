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
        filel=filel+[name+'/'+i for i in AllFile if 'all' in i]
        pass
    #文件名升序排列
    filel=sorted(filel)
    
#            try:
#            x = int(input("Please enter a number: "))
#            break
#        except ValueError:
#            print("Oops!  That was no valid number.  Try again   ")
    errorf=[]
    c=[]
    for i in filel:
        a=(int(i.split("/")[-1].split(".")[0].split("_")[-1]))
        if a>= 20150601 and a<= 20180531:
            c.append(i)
            pass
#        break
        pass
    filel=c
    #读取文件内容
    AllData=pd.DataFrame()
    a=0
    for i in filel:
        try:
            temp=pd.read_csv(i)
            temp['date']=temp.date.astype(str)
            AllData = pd.concat([AllData,temp])
#            AllData = pd.concat([AllData,pd.read_csv(i)])
            pass
        except:
            #有两个连续是空的，则读取不到文件，跳过
            if len(AllData) < 1:
                continue
                pass
            a=a+1
            #出现读取文件异常的，忽略这个文件，把已经读取的文件写入文件中，另起一个文件再读
            AllData.to_csv("./newdata_TypeConcat/"+str(a)+"AllData.csv",encoding='GBK',index=False)
            AllData=pd.DataFrame()
            errorf.append(i)
            pass
#        break
        pass
    #最后一个文件
    AllData.to_csv("./newdata_TypeConcat/"+str(a)+"AllData.csv",encoding='GBK',index=False)
    #未读取成功的
    pd.DataFrame(errorf).to_csv("errorf.csv")
    
    
    
    
    
    
    