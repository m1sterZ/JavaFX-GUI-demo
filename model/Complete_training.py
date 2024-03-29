import numpy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
import re
import random
import time
from math import sqrt

import sys

''' 外部传参 
    args[1]日志绝对路径 output/log
    args[2]训练的节点id 
'''
args = sys.argv
# output/log/node
sys.path.append(args[1] + '\\node' + args[2])
import sample as sp

# #读json 划分整个流程树所有节点的输入输出数据集
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
            # print(json_object[k])
            # print(type(json_object[k]))

            if k == key:
                if json_object[k] == "Action":
                    # print(json_object["label"])
                    self.result_list.append(json_object["label"])
                if json_object[k] == "Seq" or json_object[k] == "OrComposite":
                    if json_object[k] != "Seq":
                        self.result_list.append(json_object["label"])
                    # 只有这个部分需要增加搜索子树的按钮，搜索子树的目的是print里面的子代，若子树的某父节点是循环或其他逻辑节点，则停止搜索这个子树
                    n = "children"
                    # print(json_object[n])
                    # print(type(json_object[n]))
                    if isinstance(json_object[n], dict):
                        self.__search(json_object[n], key)
                    if isinstance(json_object[n], list):
                        for item in json_object[n]:
                            # print(item)
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

'''
def testfunc(data_recording, x_temp):
    x = torch.tensor(x_temp).float().squeeze(0)
    temp = data_recording  #获取第i个方法的计算模型/中心点
    # print('temp[0]')
    # print(temp[0])
    model_and_feature = temp[0]
    print('model_and_feature', model_and_feature)
    model_list = []
    model_feature = []
    for j in range(len(model_and_feature)):
        model_list.append(model_and_feature[j][1])
        model_feature.append(model_and_feature[j][0])
    # print(model_feature)
    temp = torch.zeros(len(model_feature), len(model_feature[0]))
    for i in range(len(model_feature)):
        temp[i,:] = torch.from_numpy(model_feature[i])
    model_feature = np.array(temp)
    print('model_feature shape', np.shape(model_feature))
    print('sample shape', np.shape(x_temp))
    dis = EuclideanDistance(x_temp, model_feature)
    print('dis shape',np.shape(dis))
    print('dis', dis) 
    dis[np.isnan(dis)] = np.nanmax(dis)
    y_pre = []
    print('model_feature', model_feature)
    loc_all = []
    for j in range(len(x_temp)):
        loc = np.where(dis[j] == np.min(dis[j]))  # 在4个模型上哪个dist最小，若为[0],则在第一个模型上式最小的
        y_pre.append(model_list[loc[0][0]](x[j]))   # 离x最近的中心点所对应的模型 对x进行预测
        loc_all.append(loc)
    y_pre_new = np.zeros((len(y_pre),len(y_pre[0])))
    for j in range(len(y_pre)):
        y_pre_new[j] = np.array(y_pre[j].detach().numpy())
    print('y_pre_new', y_pre_new)
    return y_pre_new
'''

# bp
def testfunc(data_recording, x_temp):
    # print('data_recording', data_recording[0][0][1])
    # print('x_temp', x_temp)
    data_recording = data_recording[0][0][1]
    x = torch.tensor(x_temp).float().squeeze(0)
    # print('input size', len(x[1,:]))
    # print('x', x)
    # y_pre_new = data_recording(x)
    y_pre=[]
    for j in range(len(x_temp)):
        # print(x[j])
        y_pre.append(data_recording(x[j]))
    # print('y_pre', y_pre) # 空值？
    y_pre_new = np.zeros((len(y_pre),len(y_pre[0])))
    for j in range(len(y_pre)):
        y_pre_new[j] = np.array(y_pre[j].detach().numpy())
    # print('y_pre_new', y_pre_new)
    return y_pre_new

def normalization(data):
    dimension=np.size(data)/np.size(data,1)
    for i in range(int(dimension)):
        _range = np.max(data[0,:,i]) - np.min(data[0,:,i])
        data[0, :, i]= (data[0, :, i] - np.min(data[0, :, i])) / _range
    return data


# filename='data_solution1'
# output/log/data1.json
# with open('./'+filename+'/data1.json','r',encoding='utf8')as fp:
#     data = json.load(fp)
# gkv = GetKeyValue(data)
# result_record = gkv.search_key('nodeType')
# print(result_record)

# output/log/data1.json
with open(args[1] + '\\data1.json','r',encoding='utf8')as fp:
    data = json.load(fp)
gkv = GetKeyValue(data)
# result_record = gkv.search_key('nodeType')

cal_model=[]
record=[]
main_data=[]
record_n_output=[]
data_recording=[]
combin_recording=[]
temp_record=[]
# print("result record:")
# print(result_record) # 各个节点函数
# ['com.tct.testdata.AreaMain.main(java.lang.String[])', 'com.tct.testdata.AreaMain.parseArgs(java.lang.String[])', 
# 'com.tct.testdata.AreaMain.getResult(int,double)', 'com.tct.testdata.AreaMain.normalPolygonArea(int,double)']

####### 按顺序对每个节点的数据进行获取
# print("---------")
# file_path = "C:\H\Java_codes\output\solution12_small\diagram.dot"
def read_node_names(file_path):
    res = []
    with open(file_path, 'r') as fin:
        lines = fin.read()
        line = lines.split("\"")
        cnt = 0
        for parts in line:
            if cnt % 2 == 1 and parts != ' ----> ':
                part = parts.split("\n")
                res.append(part[1])
            cnt += 1    
    # print(res)
    return res
node_names = []
# 按顺序读入所有节点，数组下标即节点id
dot_path = args[1] + '\\diagram.dot'
node_names = read_node_names(dot_path)
# print(node_names)

for j,item in enumerate(node_names):
    inp_data=[]
    oup_data=[]
    # output/log/items.txt
    # with open('./'+filename+'/'+item+'.txt') as r1_data:
    with open(args[1] + '\\' + item + '.txt') as r1_data:
        lines=r1_data.readlines()
        for i,line in enumerate(lines):
            t1,t2 = line.split("|", 1)
            num1= np.array(re.findall(r"\d+\.?\d*",t1))
            num2= np.array(re.findall(r"\d+\.?\d*",t2)) # 至少1个数字 可选一个点 n个数字 任意整数或小数  
            nnum1 = np.zeros(len(num1))
            nnum2 = np.zeros(len(num2))
            for k1 in range(len(num1)):
                nnum1[k1]=float(num1[k1])
            for k2 in range(len(num2)):
                nnum2[k2]=float(num2[k2])
            inp_data.append(nnum1)
            oup_data.append(nnum2)
    # print(item)
    # print(j)
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

# 输入和输出 data_recording[i][2], data_recording[i][3]
# print(len(data_recording[0][3]))
total_number = len(data_recording[0][2])
# print(total_number)

testing_number = int(total_number * 0.1)
data_index=[i for i in range(total_number)]
testing_index = random.sample(data_index, testing_number)  # 测试集indexes
for i in testing_index:
    data_index.remove(i)
training_index = data_index  # 训练集indexes
# a=[]
# a.append(training_index)
# a.append(testing_index)
# np.save('./'+filename+'/data_index.npy', a)
# print(training_index)

def training(i, data_recording):
    # global data_recording
    temp_recording=[]
    output_recording=[]
    # print('data_recording output', data_recording[i][3][training_index])
    # print('res', sp.train(data_recording[i][2][training_index], data_recording[i][3][training_index]))
    data_recording[i].append(sp.train(data_recording[i][2][training_index], data_recording[i][3][training_index]))
    data_recording[i].append(data_recording[i][2][training_index])
    data_recording[i].append(data_recording[i][3][training_index])
    output_recording.append(data_recording[i][3])
# args[2] 节点id
training(int(args[2]), data_recording)
print('--------train finished---------')
# print(data_recording[1][4])

def test_record(i, data_recording):
    # global data_recording
    x_loss = []
    x_dis_loss = []
    # x_temp = data_recording[len(data_recording)-1][2][testing_index]
    x_temp = data_recording[i][2][testing_index]
    # y_temp = data_recording[len(data_recording)-1][3][testing_index]
    y_temp = data_recording[i][3][testing_index]
    y = torch.tensor(y_temp).float()
    testing_pararecording = []
    testing_outputrecording = []
    # print(data_recording[i][4])
    # print('x_temp:')
    # print(x_temp)
    y_pre = testfunc(data_recording[i][4], x_temp)  #模型+输入
    # print(y_pre)
    # print(len(y_pre[0]))
    data_recording[i].append(y_pre)
    testing_outputrecording.append(y_pre)
    x_temp = y_pre
    a4=[]
    x_dis_loss = []
    for i in range(len(y_pre[1])):
        for j in range(len(y_pre)):
            # a4.append(np.sum(np.array(y[j][0]) - y_pre[j][i]))
            a4.append(np.sum(np.array(y[j][i].detach().numpy()) - y_pre[j][i]))
        x_dis_loss.append([a4])
        a4 = []
    a4_target = []
    a4_pridiction = []
    a5_target = []
    a5_pridiction = []
    for i in range(len(y_pre[1])):
        for j in range(len(y_pre)):
            a4_target.append(y[j][0].item())
            a4_pridiction.append(y_pre[j][i])
        a5_target.append([a4_target])  # y  原结果
        a5_pridiction.append([a4_pridiction])   #y_pre  实际结果
        a4_target = []
        a4_pridiction = []
    a3 = []
    # a3.append(data_recording) #所有数据
    # a3.append(training_index) #训练样本的index
    # a3.append(testing_index) #测试样本的index
    a3.append(a5_target) #预期值
    a3.append(a5_pridiction) #实际值
    a3.append(x_dis_loss) #误差
    # res_file = open('out.txt', 'a')
    # print(a3, file = res_file)
    np.save(args[1] + '\\node' + args[2] + '\\exp_result.npy', a3)
    # print('a3:', a3)
    print('------save in exp_result.npy------')

test_record(int(args[2]), data_recording)


