# -*- coding: utf-8 -*-

import pandas as pd
import time
import numpy as np
import random
import scipy.stats as ss
import math
import os
import copy
from sklearn.mixture import GaussianMixture as GMM


class DFM(object):

    # 初始化
    def __init__(self, K, G, key, value):
        self.K = K # 簇个数
        self.G = G # 高斯成分个数
        self.key = key # 对象标识字段
        self.value = value # 观测字段列表

    # 读取数据
    def load(self, data):
        self.data = data 
        # 复制数据为dataframe，并准备关于psi_1,...psi_G的初始值
        self.df = copy.deepcopy(data)
        for g in range(self.G):
            self.df['psi_%d' % g] = 0
        self.n = data[self.key].nunique() # 对象个数
        self.mids = data[self.key].unique() # 对象ID列表
        print('>> %d objects with %d observations loaded.' % (self.n, len(data)))

    # 初始化参数估计
    def init_state(self):
        # 初始化关于高斯成分的参数估计
        # 先用所有数据整体GMM（不考虑对象结构），作为初始值
        X = np.array(self.data[self.value])
        gmm = GMM(n_components=self.G).fit(X)   
        self.gmm = gmm     

        # 初始化对每个对象的系数向量phi的估计
        self.phis = {}
        df = self.df
        for mid in self.mids:
            _df = df[df[self.key]==mid]
            _probas = gmm.predict_proba(_df[self.value])
            # 对每个对象，以相应观测的后验概率之均值作为phi的初始估计
            self.phis[mid] = copy.deepcopy(_probas.mean(axis=0))
        print('>> initialization OK.')

    # 对每个观测ij，计算关于G个高斯成分的后验概率
    def est_psi(self, x):
        gmm = self.gmm 
        # 将GMM的权重临时替换为对象i的权重phi_i
        gmm.weights_ = self.phis[x[self.key]] 
        # 返回G和高斯后验概率的向量
        return gmm.predict_proba([x[self.value]])[0]

    # E-step
    def e_step(self):
        # 对每个对象的每个观测，计算后验概率 psi_ijg
        df = self.df
        df['psi'] = df.apply(lambda x: self.est_psi(x), axis=1)
        for g in range(self.G):
            # 将后验概率的向量拆解成G个单独的分量，便于后续计算
            df['psi_%d' % g] = df['psi'].apply(lambda x: x[g])

    # 辅助计算协方差矩阵
    def to_sigma(self, x):
        _x = np.array([x])
        return np.dot(_x.T, _x) 

    # M-step
    def m_step(self):
        # 更新参数
        df = self.df 

        # 1.对每个g，记录后验概率psi_ijg的累和
        psi_sums = []
        for g in range(self.G):
            psi_sums.append( df['psi_%d' % g].sum() )

        # 2.更新均值mu
        mu = self.gmm.means_
        for g in range(self.G):
            for i in range(len(self.value)):
                mu[g][i] = ( df[self.value[i]] * df['psi_%d' % g] ).sum() / psi_sums[g]

        # 3.更新方差sigma
        cov = self.gmm.covariances_
        for g in range(self.G):
            _sigmas = (df[self.value] - mu[g]).apply(lambda x: self.to_sigma(x), axis=1)
            cov[g] = (_sigmas * df['psi_%d' % g]).sum() / psi_sums[g] 

        # 4.更新phis
        for mid in self.mids:
            _df = df[df[self.key]==mid]
            for g in range(self.G):
                self.phis[mid][g] = _df['psi_%d' % g].mean()

