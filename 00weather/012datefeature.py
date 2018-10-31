#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:43:16 2018

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
    classname=pd.read_excel('./datafilldescribe/八块分.xlsx',encoding = "GBK")
    classdata=pd.DataFrame()
    classdata["point"]=classname["point"]
    classdata["part"]=classname["part"]
    tag=classname["part"]
    classtag=dict(zip(classname["point"],classname["part"]))
    tag=classname["part"].value_counts().index.tolist()
    
    tagname=[]
    for i in tag:
        tagname.append(classdata[classdata["part"]==i]["point"].tolist())
        pass
    
    rootfile="./datafill"
    newpath="./datafeature/"
    weatherdata=pd.read_csv("./weather/weatherdata.csv",encoding = "GBK")

    
    #读跟目录下所有的目录
    dirl=[]
    for paths, dirs, files in os.walk(rootfile): 
        break
        pass
    #对目录进行拼接
    cc = [rootfile+'/'+x for x in dirs]
    print("=====")
    #读取每个目录下的文件名，然后叠加
    filel=[]
    #对目录进行历边
    for name in cc:
        Paths,dirs,AllFile = file_name(name) 
        AllFile=[name+'/'+i for i in AllFile]
        #创建目录
#        a=name.split('/')[-1]
        try:
            os.mkdir(newpath+name.split('/')[-1])
        except:
            pass

        #读取每个目录下的每个文件
        for i in AllFile:
            print("dir:",name,"  ","file:",i)
            #读取文件
            #读八个区域的取平均作为每个特征
            Data=pd.read_csv(i,encoding = "GBK")
            datatemp=pd.DataFrame()
            for k,j in enumerate(tagname):
                datatemp[str(k)]=Data[j].apply(lambda x: x.mean(), axis=1)
#                break
                pass
            
            Data=pd.concat([Data.iloc[:,:3],datatemp],axis=1)

            a=newpath+name.split('/')[-1]+'/'+i.split('/')[-1]
            Data.to_csv(a,index=False,encoding = "GBK")
#            break
            pass#每个文件
        
#        break
    pass#每个目录
