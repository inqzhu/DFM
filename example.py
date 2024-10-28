# -*- coding: utf-8 -*-

import generate_mix # simulate dataset
import dfm # DFM class
import eva # evaluate clustering performance

import pandas as pd
import os 
import shutil
import numpy as np 
import time

from sklearn.cluster import KMeans

def example():
    np.random.seed(0)

    """
    0. 生成模拟数据 
    loading simulated data
    """
    source = 'simulation_data_mix'
    if os.path.exists(source):
        shutil.rmtree(source)
    os.mkdir(source)

    # 生成3类模拟数据
    # n_cluster是每类对象个数的一个基准值 
    # number of objects within 1 cluster
    n_cluster = 50
    # n_trans是每个对象观测数量的一个基准值，越大，则每个对象的观测越多，经验分布越接近真实分布 
    # number of observations for 1 object
    n_trans = 100
    ass = [ int(0.6*n_cluster), n_cluster, int(1.5*n_cluster) ]
    for i in range(1, 4):
        generate_mix.work('a'+str(i), ass[i-1], n_trans, i, source)

    """
    1. 读取生成的模拟数据，拼接所有数据为一个dataframe
    loading data
    """
    ds = []
    for filename in os.listdir(source):
        if "csv" in filename:
            d = pd.read_csv(source+'/'+filename)
            ds.append(d)
    data = pd.concat(ds, axis = 0)

    """
    2. DFM建模、迭代更新
    DFM modeling
    """
    # 指定类型/簇的数目K
    # specify number of clusters
    K = 3
    # 指定高斯成分个数G
    # specify number of Gaussian components
    G = 2
    key = 'ID' # 对象的标识，例如商户号、用户id, specify the column of the key
    value = ['x1', 'x2', 'x3'] # 特征的字段, specify the columns of the values
    worker = dfm.DFM(K, G, key, value) # 初始化 DFM, initialize DFM
    worker.load(data) # 输入数据, loading data
    worker.init_state() # 初始化参数估计, obtain initial state
    T_iter = 10 # 设置最大迭代次数, threshold of iterations

    # 开始迭代
    # training
    for t in range(T_iter):
        t0 = time.time()
        worker.e_step() 
        t1 = time.time()
        print('E step: %.2fs' % (t1-t0))

        t0 = time.time()
        worker.m_step()
        t1 = time.time()
        print('M step: %.2fs' % (t1-t0))

        print('--- Iter %d OK.' % t)

    """
    3. 基于DFM的估计结果进行聚类，并结合模拟设置中的类型评估聚类结果
    evaluate the clustering performance
    """
    y_true = [] # 模拟数据时的真实标签
    X = [] # 记录DFM对每个对象输出的向量phi, obtain vectors for all objects
    for i in range(len(data)):
        line = data.iloc[i]
        X.append( worker.phis[ line['ID'] ] )
        y_true.append( line['label']-1 )
    X = np.array(X)
    
    # 对phi 进行 K-means聚类
    # K-means clustering for vectors of objects
    clusterer = KMeans(n_clusters=K, random_state=0).fit(X)
    cluster_labels=clusterer.labels_

    # 计算聚类指标，衡量基于DFM的聚类效果
    # performance evaluation 
    [NMI, ARI, ACC] = eva.evaluate(K, cluster_labels, y_true)
    print(NMI, ARI, ACC)

# run the example
example()