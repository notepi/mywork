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
    rootfile="./newdata_TypeSpilt"
    newpath="./mergedata/"
    weatherdata=pd.read_csv("./weather/weatherdata.csv",encoding = "GBK")
#    Paths,dirs,AllFile = file_name(rootfile)
    
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
        for i in AllFile[2:]:
            #读取文件
            Data=pd.read_csv(i,encoding = "GBK")
            #将数据文件与天气文件合并
            DataAA=pd.merge(Data,weatherdata,
                            left_on=u"date",right_on=u"日期",how='inner')
            up=0
            down=0
            empty=[]
            #
            for k,j in enumerate(Data.date):
                if j not in  weatherdata[u"日期"].values:
                    up=down
                    down=k
                    #如果连续的是空，直接跳过
                    if(down-up) <= 1:
#                        print(k)
#                        print(j)
                        pass
                    else:
                        print("temp")
                        temp=Data.iloc[up:down-1,:]
                        temp=pd.merge(temp,weatherdata,
                            left_on=u"date",right_on=u"日期",how='inner')
                        del temp['hour'] 
                        #新的文件名与原文件名一直,再加上分割的起始，构造新的文件夹
                        a=newpath+name.split('/')[-1]+'/'+i.split('/')[-1].split(".")[0]+str(down)+".csv"
                        temp.to_csv(a,index=False,encoding = "GBK")
                        
#                        break
                        pass
                    pass
                pass
#            break
        
        break
        pass
    pass

#Dataaa=pd.merge(Data,weatherdata,left_on=u"date",right_on=u"日期",how='outer')