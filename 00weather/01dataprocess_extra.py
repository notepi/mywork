#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:42:26 2018

@author: pan
"""

import numpy as np
import pandas as pd
import os 

def file_name(file_dir):  
  for paths, dirs, files in os.walk(file_dir): 
#    print(paths) #当前目录路径 
#    print(dirs) #当前路径下所有子目录 
#    print(files) #当前路径下所有非目录子文件 
    pass
  return paths,dirs,files  

if __name__ == "__main__":
    Paths,dirs,AllFile = file_name("./newdata_TypeConcat")
    for i in AllFile:
        path="./newdata_TypeConcat/"+i
        
        #为每个文件分割
        a=pd.read_csv(path,encoding='GBK')
        a.date=a.date.apply(str)+a.hour.apply(str)
        #24h的关键字不在type中的内容被取出
        aa= a[a.type.apply(lambda x: "24h" not in x)]
        #提取所有特征的名字
        typename = aa.type.value_counts().index.tolist()
        #为每种创建一个目录
        for i in typename:
            try:
                os.mkdir('newdata_TypeSpilt'+"/"+i)
            except:
                pass
    #        break
            pass
        #将数据根据type重新存放
        data=[]
        for i in typename:
            data.append(aa[aa.type==i])
    #        break
            pass
        for i in range(len(data)):
            #每种数据放在目录中
            data[i].to_csv("./newdata_TypeSpilt"+"/"+typename[i]+"/"+typename[i]+path.split('/')[-1],
                            encoding='GBK',index=False)
            print(i)
    #        break
            pass
#        break
        pass
        