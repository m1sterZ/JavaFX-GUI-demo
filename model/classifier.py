import numpy
import torch
import torch.nn as nn
import numpy as np
import sample as sp
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
import re
import random
import time
from math import sqrt


def reclassify(Input, Output, model_and_feature):
    MaxLen = len(Input)
    insize = len(Input[1,:])
    outsize = len(Output[1,:])
    model_list = []
    model_feature = []
    model_feature_list = []  
    model_feature_new = []  
    model_feature_new_length = []
    # for i in range(len(model_and_feature)):
    model_list.append(model_and_feature[0])
    # for i in range(len(model_and_feature)):
    model_feature_new.append(model_and_feature[1])
    model_feature_new_length.append(model_and_feature[2])

    feature_for_iter=[]
    feature_for_iter.append([[], model_feature_new, model_feature_new_length])
    return feature_for_iter

def locate_center(Input, Output, model_and_feature, new_feature_for_iter):#最后一个参数是reclassify的返回值
    model_list = []
    len_sample=[]
    feature_list=[]
    model_list.append(model_and_feature[0])
    feature_list.append(new_feature_for_iter[0][1][0])
    len_sample.append(new_feature_for_iter[0][2][0])
    new_feature_for_iter_update = []
    for i in range(len(model_list)):
        new_feature_for_iter_update.append([feature_list[i], model_list[i], [], len_sample[i]])

    return new_feature_for_iter_update

def save_model_wrapper(net, Input, Output):
    #
    All_x = Input
    model_and_feature = []
    model_and_feature.append(net)
    model_and_feature.append(sum(All_x)/len(All_x))
    model_and_feature.append(len(All_x))
    feature_for_iter = reclassify(Input, Output, model_and_feature)
    new_feature_for_iter = locate_center(Input, Output, model_and_feature, feature_for_iter)
    model_wrapper = [new_feature_for_iter]
    return model_wrapper