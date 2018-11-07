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
#    rootfile="./real"
#    Paths,dirs,AllFile = file_name(rootfile)
#    
#    #对目录进行拼接
#    AllFile = [rootfile+'/'+x for x in AllFile]
#    print("=====")
#    #文件名升序排列
#    AllFile=sorted(AllFile)
#    AllData=pd.DataFrame()
#    for i in AllFile:
#        temp=pd.read_csv(i)
#        del temp[u"营销部名称"]
#        del temp[u"客户名称"]
#        del temp[u"客户经理"]
#        del temp[u"经营业态"]
#        del temp[u"结算方式"]
#        AllData = pd.concat([AllData,temp])
##        break
#        pass
    date=sorted(AllData[u"日期"].value_counts().index.tolist())
    DaySum=pd.DataFrame()
    for i in date:
        temp=AllData[AllData[u"日期"]==i]
        temp=pd.DataFrame(temp.iloc[:,4:].sum()).T
        DaySum=pd.concat([DaySum,temp])
#        break
        pass
    DaySum.insert(0,u"日期",date)
    DaySum.to_csv("./date/real/realDaySum.csv",encoding='GBK',index=False)
    pass
    
    
    
    
    
    
    