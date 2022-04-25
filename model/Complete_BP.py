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
import time
from math import sqrt


class GetKeyValue(object):
    def __init__(self, o):
        self.json_object = None
        self.json_object = data
        self.result_list = []
        self.deep1 = 0

    def search_key(self, key):
        self.result_list = []
        self.__search(self.json_object, key)
        return self.result_list

    def __search(self, json_object, key):

        for k in json_object:
            print(json_object[k])
            print(type(json_object[k]))

            if k == key:
                if json_object[k] == "Action":
                    print(json_object["label"])
                    self.result_list.append(json_object["label"])
                if json_object[k] == "Seq" or json_object[k] == "OrComposite":
                    if json_object[k] == "Seq":
                        print(json_object["label"])
                    else:
                        if json_object["label"].find(".getResult")==-1:
                            print(json_object["label"])
                            self.result_list.append(json_object["label"])
                        else:
                            print(json_object["label"])
                    # 只有这个部分需要增加搜索子树的按钮，搜索子树的目的是print里面的子代，若子树的某父节点是循环或其他逻辑节点，则停止搜索这个子树
                    n= "children"
                    print(json_object[n])
                    print(type(json_object[n]))
                    if isinstance(json_object[n], dict):
                        self.__search(json_object[n], key)
                    if isinstance(json_object[n], list):
                        for item in json_object[n]:
                            print(item)
                            if isinstance(item, dict):
                                self.__search(item, key)
                if json_object[k] == "Loop" or json_object[k] == "Choice" or json_object[k] == "Recurrent" or json_object[k] == "Silent":
                    print(json_object["label"])

def EuclideanDistance(x, y):
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

def testfunc(data_recording,x_temp):
    x = torch.tensor(x_temp).float().squeeze(0)
    temp = data_recording  #获取第i个方法的计算模型/中心点
    model_and_feature = temp[0]
    model_list=[]
    model_feature=[]
    for j in range(len(model_and_feature)):
        model_list.append(model_and_feature[j][1])
        model_feature.append(model_and_feature[j][0])
    model_feature=np.array(model_feature)
    print('model_feature shape', np.shape(model_feature))
    print('sample shape', np.shape(x_temp))
    dis=EuclideanDistance(x_temp,model_feature)
    print('dis shape',np.shape(dis))
    dis[np.isnan(dis)] = np.nanmax(dis)
    y_pre=[]
    print('model_feature', model_feature)
    for j in range(len(x_temp)):
        loc = np.where(dis[j] == np.min(dis[j]))  # 在4个模型上哪个dist最小，若为[0],则在第一个模型上式最小的
        y_pre.append(model_list[loc[0][0]](x[j]))   # 离x最近的中心点所对应的模型 对x进行预测
    y_pre_new=np.zeros((len(y_pre),len(y_pre[0])))
    for j in range(len(y_pre)):
        y_pre_new[j]=np.array(y_pre[j].detach().numpy())
    return y_pre_new

def testfunc_BP(data_recording,x_temp):
    x = torch.tensor(x_temp).float().squeeze(0)
    # y_pre_new = data_recording(x)
    y_pre=[]
    for j in range(len(x_temp)):
        y_pre.append(data_recording(x[j]))   # 离x最近的中心点所对应的模型 对x进行预测
    y_pre_new=np.zeros((len(y_pre),len(y_pre[0])))
    for j in range(len(y_pre)):
        y_pre_new[j]=np.array(y_pre[j].detach().numpy())

    return y_pre_new

def normalization(data):
    #print(type(data))
    #print('dataSize',np.size(data,1))
    dimension=np.size(data)/np.size(data,1)
    #print(data[0,:,0])
   # print('max:%d'%np.max(data))
    #print('dimension',dimension)
    for i in range(int(dimension)):
        _range = np.max(data[0,:,i]) - np.min(data[0,:,i])
        data[0, :, i]= (data[0, :, i] - np.min(data[0, :, i])) / _range
    return data

filename='data_solution1'
# 肖骞那边获取的json出现问题。需要把第一个子代的"nodeType"LOOP修改为"Seq"
with open('./'+filename+'/data1.json','r',encoding='utf8')as fp:
    data=json.load(fp)
gkv = GetKeyValue(data)
result_record=gkv.search_key('nodeType')
print(result_record)
cal_model=[]
record=[]
main_data=[]
record_n_output=[]
result_record.insert(len(result_record), result_record[0])
result_record.pop(0)
data_recording=[]
combin_recording=[]
temp_record=[]
####### 对每个节点的数据进行获取
for j,item in enumerate(result_record):
    inp_data=[]
    oup_data=[]
    print(item)
    print(type(item))
    with open('./'+filename+'/'+item+'.txt')as r1_data:
        lines=r1_data.readlines()
        for i,line in enumerate(lines):
            t1,t2 = line.split("|", 1)
            num1= np.array(re.findall(r"\d+\.?\d*",t1))
            num2= np.array(re.findall(r"\d+\.?\d*",t2))
            nnum1 = np.zeros(len(num1))
            nnum2 = np.zeros(len(num2))
            for k1 in range(len(num1)):
                nnum1[k1]=float(num1[k1])
            for k2 in range(len(num2)):
                nnum2[k2]=float(num2[k2])
            inp_data.append(nnum1)
            oup_data.append(nnum2)
    print(item)
    print(j)
    x = np.array(inp_data)
    y = np.array(oup_data)
    # temp1 = torch.randint(0, len(x) - 1, (1, len(x)))
    # for i in range(len(x)):
    #     temp1[0,i] = i
    # x = x[temp1]
    # x = normalization(x)
    # x = x[0]
    if y is not None:
        if min(y.shape) == 0:  #当前节点输出为空，记录当前节点的index到combin_recording
            combin_recording.append(j)
        else:  #当前节点输出非空,可以训练
            if len(combin_recording) > 0:   #前面有堆积参数
                temp_record=combin_recording
                combin_recording = []
            else:  #无堆积参数
                temp_record=[-1]
    else:  #当前节点输出为空，记录当前节点的index到combin_recording
        combin_recording.append(j)
    data_recording.append([j,temp_record,x,y])
    temp_record=[]     #data_recording  【0】当前节点的index，【1】标识符或存放的index，【2】当前节点输入参数，【3】当前节点输出参数
    # temp_record 为标识符  若为-1，表明前面的节点无堆积参数，可以直接用本节点的数据进行训练
    # temp_record 若为index，在下一个可训练的完整节点加入index对应数据作为输入
    # temp_record 若为空，需要堆积数据给下一个节点


lrate_init=1e-06
lrate_max=1e-10
temp=np.load('./'+filename+'/data_index.npy', allow_pickle=True)
training_index=temp[0]
testing_index=temp[1]
itersetting=5000000

x_temp_in = data_recording[len(data_recording)-1][2][training_index]  #模型之入参
y_temp_in = data_recording[len(data_recording)-1][3][training_index]  #模型之出参
x_temp_ou = data_recording[len(data_recording)-1][2][testing_index]  #模型之入参
y_temp_ou = data_recording[len(data_recording)-1][3][testing_index]  #模型之出参
y = torch.tensor(y_temp_ou).float()
BP_MODEL=sc.calcul_BP_learningrate_new(x_temp_in,y_temp_in,lrate_init,lrate_max,itersetting)   ################# 换网络
y_pre=testfunc_BP(BP_MODEL,x_temp_ou)

a4=[]
x_dis_loss=[]
for i in range(len(y_pre[1])):
    for j in range(len(y_pre)):
        a4.append(np.sum(np.array(y[j][i].detach().numpy()) - y_pre[j][i]))
    x_dis_loss.append([a4])
    a4 = []

a4_target=[]
a4_pridiction=[]
a5_target=[]
a5_pridiction=[]
for i in range(len(y_pre[1])):
    for j in range(len(y_pre)):
        a4_target.append(y[j][i].item())
        a4_pridiction.append(y_pre[j][i])
    a5_target.append([a4_target])  # y  原结果
    a5_pridiction.append([a4_pridiction])   #y_pre  实际结果
    a4_target = []
    a4_pridiction = []

a3 = []
a3.append(data_recording)
a3.append(training_index)
a3.append(testing_index)
a3.append(a5_target)
a3.append(a5_pridiction)
a3.append(x_dis_loss)
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
np.save('./exp_result/'+filename+'_WholeBP_model_recording_info_'+now+'.npy', a3)
