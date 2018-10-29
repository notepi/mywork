#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:43:16 2018

@author: pan
"""

import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
def file_name(file_dir):  
  for paths, dirs, files in os.walk(file_dir): 
#    print(paths) #当前目录路径 
#    print(dirs) #当前路径下所有子目录 
#    print(files) #当前路径下所有非目录子文件 
    pass
  return paths,dirs,files  

if __name__ == "__main__":
    rootfile="./mergefinal"
    
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

        #读取每个目录下的每个文件
        for i in AllFile:
            #读取文件
            Data=pd.read_csv(i,encoding = "GBK")
            a=Data.iloc[:3,:]
            break
            pass#每个文件
            # the histogram of the data
        #画图
        n, bins, patches = plt.hist(Data["values"], 30, density=True, facecolor='g', alpha=0.75)
        
        plt.figure()
        plt.xlabel('Times')
        plt.ylabel('Counts')
        plt.title('Frequency distribution')
        plt.grid(True)
        plt.show()  
    
#
#        break
        pass#每个目录
    pass

