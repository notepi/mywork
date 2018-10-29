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
    rootfile="./datafeature"
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

    #天气合并
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
            #读取文件
            Data=pd.read_csv(i,encoding = "GBK")
            
            #将数据文件与天气文件合并，若天气的没有，则标空
            DataAA=pd.merge(Data,weatherdata,
                            left_on=u"date",right_on=u"日期",how='left')
            #找到数据在天气中的对应，找到共同有的
            DataAAA=pd.merge(weatherdata,Data,
                            left_on=u"日期",right_on=u"date",how='inner')
            #判断是否融合成功
            if (len(DataAAA) != len(Data)):
                print("-------------")
                print("len DataAAA:",len(DataAAA),"len Data:",len(Data))
                print(i)
#                continue
#                break
                pass
            #判断是否融合成功
            if (len(DataAA) == 0):
                print("error!====")
                print("len DataAA:",len(DataAA),"len Data:",len(Data))
                print(i)
                continue
#                break
                pass
            
#            del DataAA['hour'] 
            del DataAA[u'日期'] 
            
            cname=DataAA.columns.values.tolist()
            #统计填充后的空值
            c=0
            for j in cname:
                c=c+len(DataAA[DataAA[j].isnull()])
#                print(len(DataAA[DataAA[j].isnull()]))
                pass
                #空缺，是nan
                DataAA[j]=DataAA[j].fillna(method="pad")
                #空缺是，""
                DataAA[j]=DataAA[j].replace("",method="pad")
            print("+++++++++++++++++:",c)
            #填充后
            c=0
            for j in cname:
                c=c+len(DataAA[DataAA[j].isnull()])
#                print(len(DataAA[DataAA[j].isnull()]))
                pass
            print("+++++++++++++++++:",c)
            #新的文件名与原文件名一直，构造新的文件夹
            a=newpath+name.split('/')[-1]+'/'+i.split('/')[-1]
            DataAA.to_csv(a,index=False,encoding = "GBK")
#            break
            pass #文件
        
#        break
        pass #目录
    
    
    
    
    rootfile="./mergedata"
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
    #城市总值合并
    citydata=pd.read_csv("./city/city.csv",encoding = "GBK")
    del citydata["hour"]
    #对目录进行历边
    for name in cc:

        typed=name.split("/")[-1]
        citytemp=citydata[citydata.type==typed]
        del citytemp["type"]
        #数据填充
        citytemp["values"]=citytemp["values"].fillna(method="pad")
#        print("empty:",len(citytemp[citytemp["values"].isnull()]))
        Paths,dirs,AllFile = file_name(name) 
        AllFile=[name+'/'+i for i in AllFile if ".DS_Store" not in i]
        
        #读取每个目录下的每个文件
        for i in AllFile:
            #读取文件
            Data=pd.read_csv(i,encoding = "GBK")
            
            #将数据文件与天气文件合并，若天气的没有，则标空
            DataAA=pd.merge(Data,citytemp,
                            left_on=u"date",right_on=u"date",how='left')
            #找到数据在天气中的对应，找到共同有的
            DataAAA=pd.merge(citytemp,Data,
                            left_on=u"date",right_on=u"date",how='inner')
            #判断是否融合成功
            if (len(DataAAA) != len(Data)):
                print("-------------")
                print("len DataAAA:",len(DataAAA),"len Data:",len(Data))
                print(i)
#                continue
#                break
                pass
            #判断是否融合成功
            if (len(DataAA) == 0):
                print("error!====")
                print("len DataAA:",len(DataAA),"len Data:",len(Data))
                print(i)
                continue
#                break
                pass

            
            cname=DataAA.columns.values.tolist()
#            #统计填充后的values仍旧有空值，继续填充
            DataAA["values"]=DataAA["values"].fillna(method="pad")

            c=0
            for j in cname:
                c=c+len(DataAA[DataAA[j].isnull()])
#                print(len(DataAA[DataAA[j].isnull()]))
                pass
            print("+++++++++++++++++:",c)
            #新的文件名与原文件名一直，构造新的文件夹
            a=newpath+name.split('/')[-1]+'/'+i.split('/')[-1]
            DataAA.to_csv(a,index=False,encoding = "GBK")
#            break
            pass #文件
        
#        break
        pass #目录
    pass
