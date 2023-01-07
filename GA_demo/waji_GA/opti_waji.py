# -*- coding: utf-8 -*-
# @Time : 06/01/23 下午 04:14
# @Author : 张斌飞
# @Email : ZhangBinFei1995@outlook.com
# @File : opti_waji.py
# @Project : pythonProject1


from GAdemo import GA
import numpy as np
import pandas as pd

Measured = pd.read_excel(' data.xlsx')
# data_frame= Measured.parse('january_2013')
data = Measured.head()
X = Measured.Measured_X
Z = Measured.Measured_Z
# print(Measured,data )
# print(X,Z)
'''   最优值为： 180 160 140 1.2 1.8  1.5'''

N_GENERATIONS = 1000  # 迭代次数
population_size = 50
chromosome_length = 25
pc = 0.8
pm = 0.09
length = 6  # 参数数目，即需要求解的未知数个数
Parameter_BOUND = [[0, 200],
                   [0, 200],
                   [0, 200],
                   [0, 2],
                   [0, 2],
                   [0, 2]
                   ]

# np.linspace(15,65+1,5)
# print(np.linspace(15,65,11))
Measured = pd.read_excel(' data.xlsx')
# data_frame= Measured.parse('january_2013')
data = Measured.head()
X = Measured.Measured_X
Z = Measured.Measured_Z
X = np.array(X)
X = np.tile(X, [population_size, 1]).T
# X = X.T
Z = np.array(Z)
Z = np.tile(Z, [population_size, 1]).T


# 自定义函数内容
def F(x):
    # print(X)
    # print(Z)

    kboomAlen = 7250  # 动臂长度
    kStickAlen = 2920  # 斗杆长度
    kBucketAlen = 2350  # 铲斗长度

    BoomJointAngle = np.linspace(15, 65, 11)
    StickJointAngle = np.linspace(-10, 10, 11)
    BucketJointAngle = np.linspace(30, 60, 11)
    # BoomJointAngle = 15
    # StickJointAngle = -10
    # BucketJointAngle =30
    error = []
    y1 = []
    y2 = []

    for i in range(11):
        # print(StickJointAngle[i])
        y = (kboomAlen + x[0]) * np.cos(BoomJointAngle[i] + x[3]) + (kStickAlen + x[1]) * np.cos(
            StickJointAngle[i] + x[4]) + \
            (kBucketAlen + x[2]) * np.cos(BucketJointAngle[i] + x[5])
        y1.append(y)
        # print(y1)
        # print(np.tile(X[i], (1,50)))
        yy = (kboomAlen + x[0]) * np.sin(BoomJointAngle[i] + x[3]) + (kStickAlen + x[1]) * np.sin(
            StickJointAngle[i] + x[4]) + \
             (kBucketAlen + x[2]) * np.sin(BucketJointAngle[i] + x[5])
        y2.append(yy)

    # y1 = np.array(y1)
    # y2 = np.array(y2)

    # print(y2.shape)
    # X = np.array(X)
    # X = np.tile(X, [population_size, 1]).T
    # # X = X.T
    # Z = np.array(Z)
    # Z = np.tile(Z, [population_size, 1]).T
    # Z = Z.T
    # print(y2 - X)
    # print(X)
    # print(X.shape)
    #
    # print(y1-X)
    # print(np.linalg.norm(y1 - X,ord=2,axis=0))
    E = np.sqrt(np.linalg.norm(y1 - X, ord=2, axis=0) + np.linalg.norm(y2 - Z, ord=2, axis=0))
    # print(E)
    return np.sqrt(E)


ga = GA(N_GENERATIONS, population_size, chromosome_length, pc, pm, length, Parameter_BOUND, F, X, Z)
ga.main()
