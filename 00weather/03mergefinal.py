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
    rootfile="./mergedata"
    newpath="./mergefinal/"
    noth=pd.read_csv("./noth.csv")
    nothtag=dict(zip(noth[u"方向"],noth["val"]))
    timelines=48
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
        AllFile=[name+'/'+i for i in AllFile if ".DS_Store" not in i]
        #创建目录
        try:
            os.mkdir(newpath+name.split('/')[-1])
        except:
            pass
        #读取每个目录下的每个文件
        finalFile=pd.DataFrame()
        for i in AllFile:
            final=pd.DataFrame()
            #读取文件
            Data=pd.read_csv(i,encoding = "GBK")
            Data["湿度"]=Data["湿度"]*100
            Data[u"风向"]=Data[u"风向"].apply(lambda x: nothtag[x])

            #文件过短
            if len(Data) < timelines:
                print(i)
                continue
                pass
#            print(i,":",len(Data))
#            final=pd.DataFrame()
            for j in range(timelines):
                temp=Data.iloc[j:j-timelines,3:-1].reset_index(drop=True)
#                print(j,"  ",j-timelines)
                final=pd.concat([final,temp],axis=1)

#                break
                pass#叠加循环
            temp=Data.iloc[j:j-timelines,:].reset_index(drop=True)
            final=pd.concat([temp.iloc[:,0:3],final,temp.iloc[:,-1]],axis=1)
            finalFile=pd.concat([finalFile,final])
#            break
            pass#每个文件

            
        del finalFile["type"]
        #新的文件名与原文件名一直，构造新的文件夹
        a=newpath+name.split('/')[-1]+"/"+name.split('/')[-1]+'.csv'
        finalFile.to_csv(a,encoding='GBK',index=False)
#        break
        pass#每个目录
    pass

