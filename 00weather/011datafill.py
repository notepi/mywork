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
    classtag=dict(zip(classname["point"],classname["part"]))
    
    rootfile="./newdata_TypeSpilt"
    newpath="./datafill/"
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
        for i in AllFile:
            print("dir:",name,"  ","file:",i)
            #读取文件
            Data=pd.read_csv(i,encoding = "GBK")
            cname=Data.columns.values.tolist()[3:]
            #在行的方式进行填充
            for j in range(len(Data)):
                #找到每一行
                temp=Data.iloc[j,:]
                #找到每个观测站
                for k in cname:
#                    k=u"延庆"
                    #判断是否为空
                    if(pd.isnull(temp[k])):
#                        print(k)
                        parter=classdata[classdata["part"]==classtag[k]]["point"].tolist()
                        #删除本身
                        parter.remove(k)
                        #历边伙伴
                        pcc=[]
                        for l in parter:
                            if pd.notnull(temp[l]):
                                #收集小伙伴的数据
                                pcc.append(temp[l])
                                pass
                            pass
                        #数据填充
                        Data[k][j]=np.mean(pcc)
                        pass
#                    break
                    pass#每个观测站
                
#                break
                pass#每行
            #统计填充后的空值
            c=0
            for j in cname:
                c=c+len(Data[Data[j].isnull()])
                pass
            print("+++++++++++++++++:",c)
            #按列填充
            for j in cname:
                #空缺，是nan
                Data[j]=Data[j].fillna(method="pad")
                #空缺是，""
                Data[j]=Data[j].replace("",method="pad")
                pass

            #新的文件名与原文件名一直，构造新的文件夹
            a=newpath+name.split('/')[-1]+'/'+i.split('/')[-1]
            Data.to_csv(a,index=False,encoding = "GBK")
#            break
            pass#每个文件
        
#        break
    pass#每个目录
