import numpy
import torch
import torch.nn as nn
import numpy as np
import sample_classify as sc
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
import re
import random
from math import sqrt
import scipy.stats as stats
import sys


def indexes_cal(target,prediction):
    #按不同的输出维度进函数~
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    # print("Errors: ", error)

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    targetDeviation = []
    targetMean = sum(target) / len(target)  # target平均值
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))

    return [squaredError,absError,targetDeviation]


## 统计结果

# 外部传参 args[0]本文件名 args[1]是节点的exp_result.npy绝对路径
# args[1] C:\H\Java_codes\output\solution12_small\node1
args = sys.argv
file_path = args[1]
model_recording = np.load(file_path + '\\exp_result.npy', allow_pickle=True)
# print(model_recording)
#[data_recording,training_index,testing_index,a5_target,a5_pridiction,x_dis_loss]
model_loss=model_recording[2]
model_indexes=[]
y_target=model_recording[0]
y_pridiction=model_recording[1]

squaredError_avg=[]
absError_avg=[]
targetDeviation_avg=[]

stat,p=stats.wilcoxon(y_target[0][0],y_pridiction[0][0], correction=True, alternative="greater")

file = open(file_path + '\\report.txt', 'w', encoding='utf8')

for i in range(len(model_loss)):
    squaredError,absError,targetDeviation = indexes_cal(y_target[i][0], y_pridiction[i][0])
    squaredError_avg.append(squaredError)
    absError_avg.append(absError)
    targetDeviation_avg.append(targetDeviation)
    print('第', i, '维预测结果对应分析指标:')    
    print("MED = ", np.median(absError))  # 中位数  absError    
    print("MAX = ", max(absError))  #   absError
    print("MIN = ", min(absError))  #   absError
    print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE  AVG  absError平均值
    print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
    print("R^2 = ", 1-((sum(squaredError) / len(squaredError))/(sum(targetDeviation) / len(targetDeviation))))  # 重相关系数

    file.write('第 ' + str(i) + ' 维预测结果对应分析指标:\n')
    file.write("MED = " + str(np.median(absError)) + '\n')
    file.write("MAX = " + str(max(absError)) + '\n')
    file.write("MIN = " + str(min(absError)) + '\n')
    file.write("MAE = " + str(sum(absError) / len(absError)) + '\n')
    file.write("MSE = " + str(sum(squaredError) / len(squaredError)) + '\n')
    file.write("RMSE = " + str(sqrt(sum(squaredError) / len(squaredError))) + '\n')
    file.write("R^2 = " + str(1-((sum(squaredError) / len(squaredError))/(sum(targetDeviation) / len(targetDeviation)))) + '\n')

    #误差分块
    # temp_abserror=absError
    # temp_abserror.sort(reverse=False)
    # class_per_1 = temp_abserror[0:int(len(absError)*0.01)]
    # class_per_5 = temp_abserror[0:int(len(absError)*0.05)]
    # class_per_10 = temp_abserror[0:int(len(absError)*0.1)]
    # class_per_30 = temp_abserror[0:int(len(absError)*0.3)]
    # class_per_50 = temp_abserror[0:int(len(absError)*0.5)]
    # class_per_80 = temp_abserror[0:int(len(absError)*0.8)]
    # class_per_100 = temp_abserror[0:int(len(absError))]
    # print("PART_1 = ", sum(class_per_1) / len(class_per_1))
    # print("PART_5 = ", sum(class_per_5) / len(class_per_5))
    # print("PART_10 = ", sum(class_per_10) / len(class_per_10))
    # print("PART_30 = ", sum(class_per_30) / len(class_per_30))
    # print("PART_50 = ", sum(class_per_50) / len(class_per_50))
    # print("PART_80 = ", sum(class_per_80) / len(class_per_80))
    # print("PART_100 = ", sum(class_per_100) / len(class_per_100))

absError=np.mean(absError_avg,axis=0)
squaredError=np.mean(squaredError_avg,axis=0)
targetDeviation=np.mean(targetDeviation_avg,axis=0)

print("平均 预测结果对应分析指标:")
print("MED = ", np.median(absError))  # 中位数  absError
print("MAX = ", max(absError))  #   absError
print("MIN = ", min(absError))  #   absError
print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE  AVG  absError平均值
print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
print("R^2 = ", 1-((sum(squaredError) / len(squaredError))/(sum(targetDeviation) / len(targetDeviation))))  # 重相关系数

file.write('\n平均预测结果对应分析指标:\n')
file.write("MED = " + str(np.median(absError)) + '\n')
file.write("MAX = " + str(max(absError)) + '\n')
file.write("MIN = " + str(min(absError)) + '\n')
file.write("MAE = " + str(sum(absError) / len(absError)) + '\n')
file.write("MSE = " + str(sum(squaredError) / len(squaredError)) + '\n')
file.write("RMSE = " + str(sqrt(sum(squaredError) / len(squaredError))) + '\n')
file.write("R^2 = " + str(1-((sum(squaredError) / len(squaredError))/(sum(targetDeviation) / len(targetDeviation)))) + '\n')

# #误差分块
# temp_abserror=absError
# temp_abserror.sort()
# class_per_1 = temp_abserror[0:int(len(absError)*0.01)]
# class_per_5 = temp_abserror[0:int(len(absError)*0.05)]
# class_per_10 = temp_abserror[0:int(len(absError)*0.1)]
# class_per_30 = temp_abserror[0:int(len(absError)*0.3)]
# class_per_50 = temp_abserror[0:int(len(absError)*0.5)]
# class_per_80 = temp_abserror[0:int(len(absError)*0.8)]
# class_per_100 = temp_abserror[0:int(len(absError))]
# print("PART_1 = ", sum(class_per_1) / len(class_per_1))
# print("PART_5 = ", sum(class_per_5) / len(class_per_5))
# print("PART_10 = ", sum(class_per_10) / len(class_per_10))
# print("PART_30 = ", sum(class_per_30) / len(class_per_30))
# print("PART_50 = ", sum(class_per_50) / len(class_per_50))
# print("PART_80 = ", sum(class_per_80) / len(class_per_80))
# print("PART_100 = ", sum(class_per_100) / len(class_per_100))

