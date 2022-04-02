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

#替换calculnet为新生成的模型函数
#215注释

#data_recording 矩阵保存模型 训练过的类
#214-243替换成自己的模型
#标识模型是对哪一个节点
# for i in range(len(data_recording)): i标识具体哪一个节点
#rnn, lstm同理， 如rnn(.....) lstm(....) 在模型代码直接补充训练部分


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
                    n= "children"
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
    # temp = torch.zeros(len(y), len(y[0]))
    # for i in range(len(y)):
    #     temp[i,:] = y[i]
    # y = np.array(temp)
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
    temp = torch.zeros(len(model_feature), len(model_feature[0]))
    for i in range(len(model_feature)):
        temp[i,:] = model_feature[i]
    model_feature = np.array(temp)
    # print('model_feature shape', np.shape(model_feature))
    # print('sample shape', np.shape(x_temp))
    dis=EuclideanDistance(x_temp,model_feature)
    # print('dis shape',np.shape(dis))
    dis[np.isnan(dis)] = np.nanmax(dis)
    y_pre=[]
    print('model_feature', model_feature)
    loc_all=[]
    for j in range(len(x_temp)):
        loc = np.where(dis[j] == np.min(dis[j]))  # 在4个模型上哪个dist最小，若为[0],则在第一个模型上式最小的
        y_pre.append(model_list[loc[0][0]](x[j]))   # 离x最近的中心点所对应的模型 对x进行预测
        loc_all.append(loc)
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
#data_solution1     # 节点 1    2    3
lrate_init=torch.tensor([0,1e-06,1e-07])
lrate_max=torch.tensor([0,1e-08,1e-07])

temp=np.load('./'+filename+'/data_index.npy', allow_pickle=True)
training_index=temp[0]
testing_index=temp[1]
 
# 注意下：肖骞那边获取的json出现问题。新数据需要需要把父代的"nodeType"里的“Loop”修改为"Seq"
with open('./'+filename+'/data1.json','r',encoding='utf8')as fp:
    data=json.load(fp)
gkv = GetKeyValue(data)
result_record=gkv.search_key('nodeType')
# print(result_record)

## 第一阶段是对每一个子计算图做训练，训练结束以后保存模型及各个模型中心点及模型参数用于后训练,
## 模型参数中心点权重等都保存在model内
## 若某个函数其使用的是对象或者只有输入无输出，先跨过这个节点，链接他的上下两个节点。因此上代输入将作为输出

cal_model=[]
record=[]
main_data=[]
record_n_output=[]
# result_record.insert(len(result_record), result_record[0])
# result_record.pop(0)
data_recording=[]
combin_recording=[]
temp_record=[]
# print(result_record)
####### 对每个节点的数据进行获取
# print("---------")
for j,item in enumerate(result_record):
    inp_data=[]
    oup_data=[]
    # print(j)
    # print(item)
    # print(type(item))
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

## 计算图训练
temp_recording=[]
output_recording=[]
print(len(data_recording))

# for i in range(len(data_recording)):
#     if len(data_recording[i][1])==0:    # data_recording[i][1]为空   需要堆积数据给下一个节点  无output_recording  针对有输入没有输出的节点
#         temp_recording.append(data_recording[i][2])
#         data_recording[i].append([])  #保持规律  data_recording  【4】节点训练好的模型， 若为空，表明当前节点无对应模型（当前节点输出空或输入输出为对象）
#         data_recording[i].append([])  #【5】训练该节点所用输入数据，
#         data_recording[i].append([])  #【6】所用输出数据
#     elif data_recording[i][1][0]<0 and i != len(result_record)-1:   # temp_record为-1 (前面的节点无堆积参数，可以直接用本节点的数据进行训练)  且  i不为末位节点（末位节点为Main）
#         #传参修改 sc.calculNet 网络， 
#         #优化器 损失函数 激活函数 学习率 层数  每一层的 输入维度 输出维度 迭代次数 
#         # data_recording[i][2][training_index] 训练集  data_recording[i][3][training_index] labels
#         data_recording[i].append(sc.calculNet_newlr(data_recording[i][2][training_index], data_recording[i][3][training_index],lrate_init[i],lrate_max[i]))
#         data_recording[i].append(data_recording[i][2][training_index])
#         data_recording[i].append(data_recording[i][3][training_index])
#         output_recording.append(data_recording[i][3])
#     elif data_recording[i][1][0]<0 and i == len(result_record)-1:  # temp_record为-1   且  i为末位节点    #Main节点的训练 1.区分测试和训练数据，2.需要捆绑第n次节点的输出结果   无需output_recording
#         if len(output_recording)==1:  # 仅存放一组
#             temp_output=output_recording[0]
#         else:   # 循环读取堆积的数据
#             temp_output = output_recording[0]
#             for j in range(len(output_recording)-1):
#                 temp_output=np.hstack((temp_output, output_recording[j+1]))
#         data_recording[i].append(sc.calculNet_newlr(np.hstack((temp_output[training_index],data_recording[i][2][training_index])), data_recording[i][3][training_index],lrate_init[i],lrate_max[i]))
#         data_recording[i].append(np.hstack((temp_output[training_index],data_recording[i][2][training_index])))  #输入由2部分构成  1， 前n次输出   2，Main 的输入
#         data_recording[i].append(data_recording[i][3][training_index])
#     else:     ## data_recording[i][1]非空 前面的节点有堆积参数，且存放的是indexes  ，需要读取并一并在该节点训练
#         temp_data=temp_recording[0]
#         data_recording[i].append(sc.calculNet_newlr(temp_data[training_index],data_recording[i][3][training_index],lrate_init[i],lrate_max[i]))
#         data_recording[i].append(temp_data[training_index])
#         data_recording[i].append(data_recording[i][3][training_index])
#         output_recording.append(data_recording[i][3])
#         temp_recording=[]

# # 只有模型输入和输出  完成完整模型的串联
# # 从输入参数开始，父代的输出参数即为子代的输入参数
# x_loss=[]
# x_dis_loss=[]
# x_temp = data_recording[len(data_recording)-1][2][testing_index]
# y_temp = data_recording[len(data_recording)-1][3][testing_index]
# y = torch.tensor(y_temp).float()
# testing_pararecording=[]
# testing_outputrecording=[]
# for i in range(len(data_recording)):
#     if len(data_recording[i][1]) == 0:
#         testing_pararecording.append(x_temp)
#         y_pre=x_temp
#         data_recording[i].append([])
#     elif data_recording[i][1][0]<0 and i != len(result_record)-1:
#         y_pre=testfunc(data_recording[i][4],x_temp)  #模型+输入
#         data_recording[i].append(y_pre)
#         testing_outputrecording.append(y_pre)
#     elif data_recording[i][1][0]<0 and i == len(result_record)-1:  #Main 1. 区分测试和训练数据，2. 需要捆绑第n次节点的输出结果
#         if len(testing_outputrecording)==1:  # 仅存放一组
#             temp_trainingoutput=testing_outputrecording[0]
#         else:   # 循环读取堆积的数据
#             temp_trainingoutput = testing_outputrecording[0]
#             for j in range(len(testing_outputrecording)-1):
#                 temp_trainingoutput=np.hstack((temp_trainingoutput, testing_outputrecording[j+1]))
#         y_pre=testfunc(data_recording[i][4],np.hstack((temp_trainingoutput, data_recording[len(data_recording) - 1][2][testing_index])))
#         data_recording[i].append(y_pre)
#     else:
#         temp_data=testing_pararecording[0]
#         y_pre = testfunc(data_recording[i][4],temp_data)
#         testing_pararecording=[]
#         data_recording[i].append(y_pre)
#         testing_outputrecording.append(y_pre)
#     x_temp=y_pre    # 将输出加载为下一个节点的输入

# # 最后一个模型Main的输出y_pre即为完整计算图的输出结果。
# # 基于模型输出和完整计算图的输出计算误差。

# a4=[]
# x_dis_loss=[]
# for i in range(len(y_pre[1])):
#     for j in range(len(y_pre)):
#         a4.append(np.sum(np.array(y[j][i].detach().numpy()) - y_pre[j][i]))
#     x_dis_loss.append([a4])
#     a4 = []

# a4_target=[]
# a4_pridiction=[]
# a5_target=[]
# a5_pridiction=[]
# for i in range(len(y_pre[1])):
#     for j in range(len(y_pre)):
#         a4_target.append(y[j][i].item())
#         a4_pridiction.append(y_pre[j][i])
#     a5_target.append([a4_target])  # y  原结果
#     a5_pridiction.append([a4_pridiction])   #y_pre  实际结果
#     a4_target = []
#     a4_pridiction = []

# a3 = []
# a3.append(data_recording) #所有数据
# a3.append(training_index) #训练样本的index
# a3.append(testing_index) #测试样本的index
# a3.append(a5_target) #预期值
# a3.append(a5_pridiction) #实际值
# a3.append(x_dis_loss) #误差
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
# np.save('./exp_result/'+filename+'_CalculNet_model_recording_info_'+now+'.npy', a3)
