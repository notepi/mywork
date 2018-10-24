#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:36:20 2018

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
#    Data = pd.read_excel('北京小时天气数据201506-201805.xlsx',encoding = "GBK")
#    Data.to_csv("北京小时天气数据201506-201805.csv",index=False,encoding = "GBK")
    Data=pd.read_csv("北京小时天气数据201506-201805.csv",encoding = "GBK")
    a=Data['日期'].apply(lambda x:x.split('-')[0]+x.split('-')[1]+x.split('-')[2])
    b=Data['时间'].apply(lambda x:str(int(x.split(':')[0])))
    a=a+b
    a.name="日期"
    b=Data["气温"].apply(lambda x:x.split('.')[0])
    c=Data["露点"].apply(lambda x:x.split('.')[0])
    d=Data["湿度"]
    e=Data["气压"].apply(lambda x:int(x[:-2])-1010)
    f=Data["风向"]
    g=Data["风速"].apply(lambda x:x.split('/')[0][:-2])
    
    final=pd.concat([a,b,c,d,e,f,g],axis=1)
    
    name=final.columns.values.tolist()
    for i in name[1:]:
        #用前一个数据进行填充
        print(i)
        #空缺，是nan
        final[i]=final[i].fillna(method="pad")
        #空缺是，""
        final[i]=final[i].replace("",method="pad")
#        break
        pass
    final.to_csv("weatherdata.csv",index=False,encoding = "GBK")
    
    
    pass
