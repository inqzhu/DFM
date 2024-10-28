#!/usr/bin/python2.7 
# -*- coding: utf-8 -*-

# 生成模拟数据
import random
import pandas as pd
import scipy.stats as ss
import os, shutil
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

def work(mark, N, baseCount, flag, gd):
    # 生成模拟观测笔数，每个对象对应多个交易
    count = np.random.poisson(baseCount,size=N)
    dataSet = []

    # 生成每笔交易金额
    for i in range(0, N):
        temp_set = []
        name = mark + '_' + str(i)

        # 单个商户交易笔数
        count_trans = int(count[i])

        _mean = [0, 0, 0]
        _cov = [
        [1, 0.5, 0.2], 
        [0.5, 1, 0.5],
        [0.2, 0.5, 1]
        ]
        gmm1 = multivariate_normal(mean = _mean, cov = _cov)
       

        _mean = [0, 0, 0]
        _cov = [
        [1, -0.5, -0.2], 
        [-0.5, 1, -0.5],
        [-0.2, -0.5, 1]
        ]
        gmm2 = multivariate_normal(mean = _mean, cov = _cov)
       

        if flag == 1:
            data1 = gmm1.rvs(int(count_trans*0.2)+1)
            data2 = gmm2.rvs(int(count_trans*0.8)+1)
            for i in range(len(data1)):
                dataSet.append([name, 
                    data1[i][0], data1[i][1], data1[i][2],
                    flag ])
            for i in range(len(data2)):
                dataSet.append([name, 
                    data2[i][0], data2[i][1], data2[i][2],
                    flag ])

        elif flag == 2:
            data1 = gmm1.rvs(int(count_trans*0.5)+1)
            data2 = gmm2.rvs(int(count_trans*0.5)+1)
            for i in range(len(data1)):
                dataSet.append([name, 
                    data1[i][0], data1[i][1], data1[i][2],
                    flag ])
            for i in range(len(data2)):
                dataSet.append([name, 
                    data2[i][0], data2[i][1], data2[i][2],
                    flag ])

        elif flag == 3:
            data1 = gmm1.rvs(int(count_trans*0.8)+1)
            data2 = gmm2.rvs(int(count_trans*0.2)+1)
            for i in range(len(data1)):
                dataSet.append([name, 
                    data1[i][0], data1[i][1], data1[i][2],
                    flag ])
            for i in range(len(data2)):
                dataSet.append([name, 
                    data2[i][0], data2[i][1], data2[i][2],
                    flag ])

    print("-->: generating finished")

    dataSet = pd.DataFrame(dataSet, columns = ['ID', 'x1', 'x2', 'x3', 'label'])
    
    print("= [%d] - [%.2f]-[%.2f] [%.2f]-[%.2f]" % (flag, dataSet['x1'].mean(),
        dataSet['x1'].std(), dataSet['x2'].mean(),
        dataSet['x2'].std()))

    dataSet.to_csv(gd+'/'+mark+'.csv',index=False)
    return dataSet[ ['x1', 'x2', 'x3'] ]



