import torch
import torch.nn as nn
import numpy as np
#import  sample_classify  as sc
import matplotlib.pyplot as plt
import math
import os
import time
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

#input 训练集， output == label
def calculNet_newlr(Input, Output,lrate,lrate_max):
    #Input=np.load('Input.npy')
    #Output=np.load('Output.npy')
    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])
    model_and_feature = []

    batchSize = 64
    runsize = 1
    groupsize = 1
    evaluated = 2000
    Allloss = []
    net_id = 0
    losslist = []
    losstemp = []
    loss_for_node = np.array([])
    model = []
    optimer = []
    loss_func = []
    X_evaluate = []
    Y_evaluate = []
    sc_er = sample_classify()  # sample_classifier 类
    class_position_list = []
    lr_list = []
    # lrate = 1e-06
    for i in range(groupsize):
        #初始化模型
        model.append(Classify_Net(insize, ousize, net_id))  # 生成一个group中的模型,并赋予id
        model_par = model[i].state_dict()
        model_bais = model_par['Base.weight']
        t1 = model_bais.detach().numpy()
        print("t1", t1)

        net_id = net_id + 1
        optimer.append(torch.optim.SGD(model[i].parameters(), lr=lrate))

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimer[i], mode='min', factor=0.95, patience=1000, verbose=False,
        #                                            threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                            eps=lrate_max)

        losstemp=[]   # 记录在一次实验（run）中每个模型在每一次迭代的损失
        loss_func.append(nn.SmoothL1Loss())
        for epoch in range(evaluated):
            optimer[i].zero_grad()
            # 选择数据并进行处理，每一次迭代选择之后的数据都应该被保存起来
            seed = torch.randint(0, MaxLen - 1, (1, batchSize))
            x = Input[seed]
            y = Output[seed]

            x = torch.tensor(x).float().squeeze(0)
            y = torch.tensor(y).float()
            out = model[i](x)
            loss = loss_func[i](out, y)  # 计算误差

            loss.backward()
            optimer[i].step()
            # scheduler.step(loss) ##### 第二类
            # lr_list.append(optimer[i].state_dict()['param_groups'][0]['lr'])  #####

            losstemp.append(loss)

            model_par = model[i].state_dict()
            model_bais = model_par['Base.weight']
            t2 = model_bais
            # print('t2', t2.detach().numpy())
            if epoch % 1024 == 0:
                print('groups={0}, epoch{1}'.format(i + 1, epoch))
                print('loss',loss)
                # print('current_lrate',lr_list[len(lr_list)-1])
                print('t2', t2[0].detach().numpy())

        # plt.plot(range(len(losstemp)), losstemp, color='r')
        # plt.show()
        # plt.savefig("filename.png")

        # Allloss.(losstemp)
        # print(type(losstemp[i]))
        seed = torch.randint(0, MaxLen - 1, (1, int(MaxLen / 10)))
        x = Input[seed]
        y = Output[seed]

        All_x = torch.tensor(x).float().squeeze(0)
        All_y = torch.tensor(y).float().squeeze(0)
        All_out = model[i](All_x)  #训练好的模型model 使用新的数据集all_x进行验证，获得验证标签

        loss_for_node = np.abs((All_out - All_y).detach().numpy())
        loss_for_node = np.sum(loss_for_node, axis=1) / ousize
        loss_for_node = np.resize(loss_for_node, (np.shape(loss_for_node)[0], 1)) #获得节点误差--获得初始模型误差

        sc_er.class_feature(All_x.detach().numpy(), All_y.detach().numpy(), loss_for_node, 3, insize, ousize, model,
                            model[i].id, 0, method='percent') #确定子代节点，依据损失确定子代节点的误差
        feature_list = sc_er.feature_list

    net_id = 0
    model_list = []
    optimer_list = []
    loss_fun_list = []
    model_and_feature = []
    loss_list = []
    seed = torch.randint(0, MaxLen - 1, (1, MaxLen))
    X_sample = Input[seed]
    Y_sample = Output[seed]

    #lrate=0.000005
    classify_net_main_epoch=30000
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    classify_net_main(insize, ousize, net_id, lrate, X_sample[0], Y_sample[0], 4, 3, classify_net_main_epoch, sc_er, model_list,
                      optimer_list, loss_fun_list, -1, model_and_feature, loss_list, 2,lrate_max,now)
    # print('sc_er.model_list',sc_er.model_list)
    # print('sc_er.detth',sc_er.depth_list)
    print(model_and_feature)
    for i in range(len(model_and_feature)):
        model_par1 = model_and_feature[i][0].state_dict()
        model_bais1 = model_par1['Base.weight']
        t3 = model_bais1
        print('t3', t3.detach().numpy())
    # np.save('model_and_feature2000.npy', model_and_feature)  # 根据样本误差形成不同的节点保存到model_and_feature.npy
    # np.save('sc_er.npy', [sc_er.depth_list, loss_list])
    # np.save('./model_save/model_and_feature'+now+'.npy', model_and_feature)
    np.save('./model_save/model_and_feature{0}-classify_net_main_epoch{1}.npy'.format(now, classify_net_main_epoch), model_and_feature)
    # np.save('./model_save/sc_er'+now+'.npy', [sc_er.depth_list, loss_list])
    new_feature_for_iter=reclassify(Input, Output,model_and_feature,[sc_er.depth_list, loss_list],1)  #末尾times
    new_feature_for_iter_update=locate_center(Input, Output,model_and_feature,new_feature_for_iter)
    abcandc=[new_feature_for_iter_update]
    return abcandc



def reclassify(Input,Output,model_and_feature,sc_er,tttimes):
    #tttimes=10
    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])
    # model_and_feature = []
    print('type:', type(Input))
    print('inputsize:', np.shape(Input))
    print('inputsize:', np.shape(Input[[1, 3, 6]]))
    depth_list = sc_er[0]
    model_list = []
    model_feature = []
    model_feature_list=[]  # old recording
    model_feature_new=[]  # new recording  dataset
    model_feature_new_length=[]
    for i in range(len(model_and_feature)):
        model_list.append(model_and_feature[i][0])
    for i in range(len(model_and_feature)):
        # model_feature.append(model_and_feature[i][1])
        # temp=sum(model_and_feature[i][1]) / len(model_and_feature[i][1])
        # model_feature_list.append(temp[0:len(temp)-1])
        model_feature_new.append(model_and_feature[i][2])
        model_feature_new_length.append(model_and_feature[i][3])

    feature_for_iter=[]
    feature_for_iter.append([[], model_feature_new, model_feature_new_length])    #################### buqu chong jisuan zhongxindian
    return feature_for_iter

def locate_center(Input,Output,model_and_feature,new_feature_for_iter):
    model_list = []
    len_sample=[]
    feature_list=[]
    for i in range(len(model_and_feature)):
        model_list.append(model_and_feature[i][0])
        feature_list.append(new_feature_for_iter[0][1][i])
        len_sample.append(new_feature_for_iter[0][2][i])
    new_feature_for_iter_update=[]
    for i in range(len(model_list)):
        new_feature_for_iter_update.append([feature_list[i], model_list[i], [], len_sample[i]])

    return new_feature_for_iter_update

def calculNet_simplify(Input, Output,lrate,lrate_max,evaluated):
    # Input = np.load('data_solution_new_1_1/Input.npy')
    # Output = np.load('Output.npy')
    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])
    model_and_feature = []

    batchSize = 64
    runsize = 1
    groupsize = 1
    # evaluated = 20000
    Allloss = []
    net_id = 0
    losslist = []
    losstemp = []
    loss_for_node = np.array([])
    model = []
    optimer = []
    loss_func = []
    X_evaluate = []
    Y_evaluate = []
    sc_er = sample_classify()
    class_position_list = []
    lr_list = []
    # lrate = 1e-06
    for i in range(groupsize):
        #初始化模型
        model.append(Classify_Net(insize, ousize, net_id))  # 生成一个group中的模型,并赋予id
        model_par = model[i].state_dict()
        model_bais = model_par['Base.weight']
        t1 = model_bais.detach().numpy()
        print("t1", t1)  #模型权重

        net_id = net_id + 1
        optimer.append(torch.optim.SGD(model[i].parameters(), lr=lrate))  # 可迭代字典

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimer[i], mode='min', factor=0.95, patience=1000, verbose=False,
        #                                            threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                            eps=lrate_max)

        losstemp=[]
        loss_func.append(nn.SmoothL1Loss())
        for epoch in range(evaluated):
            optimer[i].zero_grad()
            # 选择数据并进行处理，每一次迭代选择之后的数据都应该被保存起来
            seed = torch.randint(0, MaxLen - 1, (1, batchSize))
            x = Input[seed]
            y = Output[seed]

            x = torch.tensor(x).float().squeeze(0)
            y = torch.tensor(y).float()
            out = model[i](x)
            loss = loss_func[i](out, y)  # 计算误差

            loss.backward()
            optimer[i].step()
            # scheduler.step(loss) ##### 第二类
            # lr_list.append(optimer[i].state_dict()['param_groups'][0]['lr'])  #####

            losstemp.append(loss)

            model_par = model[i].state_dict()
            model_bais = model_par['Base.weight']
            t2 = model_bais
            # print('t2', t2.detach().numpy())
            if epoch % 1024 == 0:
                print('groups={0}, epoch{1}'.format(i + 1, epoch))
                print('loss',loss)
                # print('current_lrate',lr_list[len(lr_list)-1])
                print('t2', t2[0].detach().numpy())

        # plt.plot(range(len(losstemp)), losstemp, color='r')
        # plt.show()
        # plt.savefig("filename.png")

        # Allloss.(losstemp)
        # print(type(losstemp[i]))
        seed = torch.randint(0, MaxLen - 1, (1, int(MaxLen / 10)))
        x = Input[seed]
        y = Output[seed]

        All_x = torch.tensor(x).float().squeeze(0)
        All_y = torch.tensor(y).float().squeeze(0)
        All_out = model[i](All_x)  #训练好的模型model 使用新的数据集all_x进行验证，获得验证标签

        loss_for_node = np.abs((All_out - All_y).detach().numpy())
        loss_for_node = np.sum(loss_for_node, axis=1) / ousize
        loss_for_node = np.resize(loss_for_node, (np.shape(loss_for_node)[0], 1)) #获得节点误差--获得初始模型误差

        sc_er.class_feature(All_x.detach().numpy(), All_y.detach().numpy(), loss_for_node, 3, insize, ousize, model,
                            model[i].id, 0, method='percent') #确定子代节点，依据损失确定子代节点的误差
        feature_list = sc_er.feature_list

    net_id = 0
    model_list = []
    optimer_list = []
    loss_fun_list = []
    model_and_feature = []
    loss_list = []
    seed = torch.randint(0, MaxLen - 1, (1, MaxLen))
    X_sample = Input[seed]
    Y_sample = Output[seed]

    #lrate=0.000005
    classify_net_main_epoch=20000
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    classify_net_main(insize, ousize, net_id, lrate, X_sample[0], Y_sample[0], 4, 3, classify_net_main_epoch, sc_er, model_list,
                      optimer_list, loss_fun_list, -1, model_and_feature, loss_list, 2,lrate_max,now)
    # print('sc_er.model_list',sc_er.model_list)
    # print('sc_er.detth',sc_er.depth_list)
    print(model_and_feature)
    for i in range(len(model_and_feature)):
        model_par1 = model_and_feature[i][0].state_dict()
        model_bais1 = model_par1['Base.weight']
        t3 = model_bais1
        print('t3', t3.detach().numpy())
    # np.save('model_and_feature2000.npy', model_and_feature)  # 根据样本误差形成不同的节点保存到model_and_feature.npy
    # np.save('sc_er.npy', [sc_er.depth_list, loss_list])
    # np.save('./model_save/model_and_feature'+now+'.npy', model_and_feature)
    np.save('./model_save/model_and_feature{0}-classify_net_main_epoch{1}.npy'.format(now, classify_net_main_epoch), model_and_feature)
    new_feature_for_iter=reclassify_simplify(Input, Output,model_and_feature,[sc_er.depth_list, loss_list],10)  #末尾times
    new_feature_for_iter_update=locate_center_simplify(Input, Output,model_and_feature,new_feature_for_iter)
    abcandc=[new_feature_for_iter_update]

    return abcandc

def reclassify_simplify(Input,Output,model_and_feature,sc_er,tttimes):
    #tttimes=10
    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])
    # model_and_feature = []
    print('type:', type(Input))
    print('inputsize:', np.shape(Input))
    print('inputsize:', np.shape(Input[[1, 3, 6]]))
    depth_list = sc_er[0]
    model_list = []
    model_feature = []
    for i in range(len(model_and_feature)):
        model_list.append(model_and_feature[i][0])
    for i in range(len(model_and_feature)):
        model_feature.append(model_and_feature[i][1])

    loss_func = nn.SmoothL1Loss()

    # seed = torch.linspace(0, MaxLen - 1, int(MaxLen/20)).int().reshape(1, int(MaxLen/20))
    # 进行10次随机中心点计算并保存其信息
    feature_for_iter = []
    for times in range(tttimes):
        loss_list = []
        seed = torch.randint(0, MaxLen - 1, (1, 5000))
        x = Input[seed]
        y = Output[seed]
        # x = normalization(x)
        x = torch.tensor(x).float().squeeze(0)
        y = torch.tensor(y).float()

        for i in range(len(model_list)):
            # 计算样本在每个模型上面的时候得到的误差，使用距离这个实数值表示所有输出维度在一个模型上的结误差
            # print(y)
            loss_in_model = model_list[i](x)
            # print('out_size:',out.size())
            loss_numpy = np.abs((loss_in_model - y).detach().numpy()[0])
            print('shape of loss_numpy', np.shape(loss_numpy))
            samplesize = np.shape(loss_numpy)[0]
            print('times:{0},model:{1}/{2} '.format(times, i, len(model_list)))
            print('loss_numpy shape',
                  np.shape(np.dot(loss_numpy, loss_numpy.T)[np.arange(samplesize), np.arange(samplesize)]))
            loss_list.append(np.dot(loss_numpy, loss_numpy.T)[np.arange(samplesize), np.arange(samplesize)])

        # 把样本数据在所有的模型上面的误差进行总结
        loss_list = np.array(loss_list)
        print('loss_list type:', type(loss_list))
        print('loss_list shape', np.shape(loss_list))
        #np.save('loss_for_everymodel.npy', loss_list)
        sampleminindex = []
        loss_list = np.array(loss_list)
        print('lenlosslist', len(loss_list))
        # 对每个样本寻找最优的分类
        for i in range(np.shape(loss_list)[1]):
            # 判断每个样本在哪个位置的时候误差最小。从0开始
            sampleminindex.append(np.argmin(loss_list[:, i]))
        feature_list = []
        class_index = []
        new_feature_list = []
        print('sample min index:', sampleminindex)
        sampleminindex = np.array(sampleminindex)
        for i in range(len(model_list)):
            # classindex[i]就是第i个模型的样本在x中的位置，i从0开始
            class_index.append(np.where(sampleminindex == i))
            print('class_index[', i, ']"', class_index[i])
            class_i_sample = np.array(x[class_index[i]])
            print('class', i, 'sample:', class_i_sample)
            if len(class_i_sample.shape) > 1:
                class_i_feature = np.sum(class_i_sample, axis=0) / len(class_i_sample)  # 选择第i个模型的数据的特征值计算   抛出异常RuntimeWarning: invalid value encountered in true_divide  因为部分节点是nan
            elif len(class_i_sample.shape) == 1:
                class_i_feature = class_i_sample
            else:
                print('sample_classify 326 error')
            # feature_list i 就是第i个model
            print('model[', i, '] feature:', class_i_feature)
            new_feature_list.append(class_i_feature)
        feature_for_iter.append([seed, new_feature_list, class_index])
    #np.save('new_feature_for_iter', feature_for_iter)
    return feature_for_iter

def locate_center_simplify(Input,Output,model_and_feature,new_feature_for_iter):
    model_list = []
    model_curve=[]
    for i in range(len(model_and_feature)):
        model_list.append(model_and_feature[i][0])
        # model_curve.append(model_and_feature[i][4])
    #[seed,new_feature_list,class_index]
    #seed是本次迭代选择的样本的位置
    #new_feature_list是对应的每一个模型的在本次计算中的中心点
    #classindex[i]就是第i个模型的样本在x中的位置
    new_feature_for_iter_update=new_feature_for_iter
    # iter 表示迭代的总次数
    iter = len(new_feature_for_iter_update)
    model_num = len(new_feature_for_iter_update[0][2])
    model_sample = []
    for j in range(model_num):
        model_sample.append([])

    print('iter', iter)

    # i表示第i次迭代
    for i in range(iter):
        seed = new_feature_for_iter_update[i][0]  # 之前保存的数据index
        seed = np.array(seed)[0]
        # print('seed',seed)
        class_index = new_feature_for_iter_update[i][2]  # 之前保存的第1组运行下，每个模型对应的数据在5000个数据中的index
        for j in range(model_num):
            model_sample[j].append(seed[class_index[j]])  # 对每个模型sample   去找5000个数据中的index在实际100w次数据下的index

    # 去除对于多余的样本
    for j in range(model_num):
        temp = []
        for i in range(len(model_sample[j])):
            a = model_sample[j][i]
            temp = list(set(temp) ^ set(a))  # set(temp)^set(a)对temp和a去重，去重后格式为字典，将其转化为list   set ^set:前面的集合去掉后面的集合剩下的部分
        model_sample[j] = temp  # 10run个list的选择结果去重
        print(temp)
    # 重新对模型的中心点进行划分
    new_feature_for_iter_update = []
    model_list_update = []
    new_feature_record=[]
    len_record=np.zeros(model_num)
    for i in range(model_num):
        x = Input[model_sample[i]]
        y = Output[model_sample[i]]
        x = torch.tensor(x).float().squeeze(0)
        y = torch.tensor(y).float()
        x = np.array(x)
        y = np.array(y)
        #print('len(x)', x.shape[0])
        # feature_list i 就是第i个model
        if len(x.shape)>1:
            len_record[i] = len(x)
            class_i_feature = np.sum(x, axis=0) / len(x)
        elif len(x.shape)==1:
            len_record[i] = 1
            class_i_feature = x
        else:
            print('sample_classify 326 看看len还有啥问题呢')

        print('model[', i, '] feature:', class_i_feature)
        new_feature_record.append([class_i_feature, model_list[i],model_sample[i], len(x)])
    np.save('new_feature_record.py',new_feature_record)
    nums=len_record
    sorted_nums = sorted(enumerate(nums), key=lambda x: x[1])
    idx = [i[0] for i in sorted_nums]
    nums = [i[1] for i in sorted_nums]
    idx_inverse_record = np.zeros(len(len_record))
    nums_inverse_record = np.zeros(len(len_record))
    temp1=0
    for i in range(len(len_record),0,-1):
        idx_inverse_record[temp1]=idx[i-1]
        nums_inverse_record[temp1]=nums[i-1]
        temp1=temp1+1
    if len(np.flatnonzero(nums_inverse_record)) > int(model_num*0.50):
        for i in range(int(model_num*0.50)):
            new_feature_for_iter_update.append(new_feature_record[int(idx_inverse_record[i])])
            model_par2 = new_feature_record[int(idx_inverse_record[i])][1].state_dict()
            model_bais2 = model_par2['Base.weight']
            t4 = model_bais2
            print('t4', t4.detach().numpy())
    else:
        for i in range(len(np.flatnonzero(nums_inverse_record))):
            new_feature_for_iter_update.append(new_feature_record[int(idx_inverse_record[i])])
            model_par2 = new_feature_record[int(idx_inverse_record[i])][1].state_dict()
            model_bais2 = model_par2['Base.weight']
            t4 = model_bais2
            print('t4', t4.detach().numpy())

    return new_feature_for_iter_update



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


#node表示的是在树中的一个节点
class node:
    def __init__(self,feature=[],id=-1,depth=-1):
        self.feature=feature
        self.children_num=0
        self.id=id
        #父节点只保存子节点的id，不保存实体
        self.children_id=[]
        self.net=[]
    def change_feature(self,feature):
        self.feature=feature
    def add_child(self,feature,id):
        temp=node(feature,id)
        self.children_num=self.children_num+1
        self.children_id.append(id)
        return  temp
    def set_net(self,net):
        self.net=[]
        self.net.append(net)

class sample_classify:
    #一个类对样本进行分类，并保存每个样本分类的特征，并对输入的样本根据特征进行分类
    #目前采取的分类标准是按照，根据误差值的百分比，比如分为三类，则误差前20%的样本作为一类，误差后20%的样本作为一类，中间的60%作为一个分类，也可以根据上下限之间的间距选择前30%，30%-70%
    #70%-100%为三个分类。
    #之后选择分类的中心点作为该类的特征
    #分类是分层的，最终会生成一个树形结构
    #所有的特征使用numpy的数组表示
    #初始研究为了简便假设每次分类都分成三类

    def __init__(self):
        #node 对象只负责保存
        self.node_list=[] #和id一一对应保存全部的特征对象，方便根据id选择对象
        self.feature_list = []#保存每个id对应的feature
        self.node=node(id=0) #创建一个node类，保存每一层的特征，每一个样本分类作为了一个node
        #class_id唯一标识分类
        self.class_id=1
        #保存每个model的
        self.depth_list=[]
        self.model_list=[]
    def update_id(self,node):
        self.class_id=self.class_id+1
        self.node_list.append(node)
        self.feature_list.append(node.feature)
    #参数包含了X，Y分类数量，以及选择对样本分类的方法，返回值是对样本的分类。返回的是样本分类序列，分类都是当前分类树的最终叶子结点

    def locat_node_by_id(self,id):
        node_len=len(self.node_list)
        for i in range(node_len):
            if self.node_list[i].id==id:
                return self.node_list[i]
        print('没有该id对应的分类')
        return []
    # ss_num需要分类的数量，insize，ousize输入维度和输出维度，id: 父节点的id
    #method选择哪一种方法
    #函数实现了对样本的分类，并更新node_list和feature_list。在训练网络的时候可以把训练好的模型保存在node当中
    #model表示要被保存的模型
    #X,Y,Loss。 Cla
    #该函数只负责计算class_feature
    #  依据loss分类数据，计算各组数据特征，分配给各数据所在节点
    def class_feature(self,X,Y,Loss,class_num,insize,outsize,model,id,depth,method='percent'):
        if method=='percent':
            cbsample=np.hstack((X,Y))
            cbsample = np.hstack((cbsample, Loss))
            cbsample=np.array(cbsample)
            # print('cbsample size:', np.shape(cbsample))
            #print(Loss[0][0])
            #print(cbsample[0][10])
            sample_len=len(X)
            # print('lenX',sample_len)
            #根据排序后的顺序选择样本的类别。
            sampleseq=np.lexsort(cbsample.T)
            # print('X',X)
            # print('Y',Y)
            # print(cbsample[sampleseq][:,len(cbsample[1,:])-1])  #检查，数据【x,y】按照实现误差从小到大排序

            Per_20=int(0.2*sample_len)
            Per_60=int(0.6*sample_len)
            Per_33 = int(0.33 * sample_len)

            #划分
            class_one=cbsample[sampleseq[0:Per_20]]
            class_two=cbsample[sampleseq[Per_20:Per_20+Per_60]]
            class_three=cbsample[sampleseq[Per_20+Per_60:sample_len-1]]
            #去loss
            class_one=np.array(class_one)[:,0:insize+outsize]
            class_two=np.array(class_two)[:,0:insize+outsize]
            class_three=np.array(class_three)[:,0:insize+outsize]
            #对数据每个维度求平均值（不同误差所在class区域数据确实有一些差异）
            if len(class_one.shape) > 1:
                class_one_feature = np.sum(class_one,axis=0)/len(class_one)  # 选择第i个模型的数据的特征值计算   抛出异常RuntimeWarning: invalid value encountered in true_divide  因为部分节点是nan
            elif len(class_one.shape) == 1:
                class_one_feature = class_one
            else:
                print('sample_classify 510 看看1组中心点还有啥问题呢')
            if len(class_two.shape) > 1:
                class_two_feature = np.sum(class_two,axis=0)/len(class_two)  # 选择第i个模型的数据的特征值计算   抛出异常RuntimeWarning: invalid value encountered in true_divide  因为部分节点是nan
            elif len(class_two.shape) == 1:
                class_two_feature = class_two
            else:
                print('sample_classify 326 看看2组中心点还有啥问题呢')
            if len(class_three.shape) > 1:
                class_three_feature = np.sum(class_three,axis=0)/len(class_three)  # 选择第i个模型的数据的特征值计算   抛出异常RuntimeWarning: invalid value encountered in true_divide  因为部分节点是nan
            elif len(class_three.shape) == 1:
                class_three_feature = class_three
            else:
                print('sample_classify 326 看看3组中心点还有啥问题呢')

            #根据id选择对应的节点对象
            #在次训练的时候保存model和depth信息
            if id==0:
                run_node=self.node
                run_node.net.append(model)
                self.model_list.append(model)
                self.depth_list.append(depth)
            else:
                run_node = self.locat_node_by_id(id)
                run_node.net.append(model)
                self.model_list.append(model)
                self.depth_list.append(depth)
            #添加新node到nodelist，给定id
            temp=run_node.add_child(class_one_feature,self.class_id)
            self.update_id(temp)
            temp=run_node.add_child(class_two_feature, self.class_id)
            self.update_id(temp)
            temp=run_node.add_child(class_three_feature, self.class_id)
            self.update_id(temp)

        elif method=='value':
            pass
        else:
            print('error class')

    #返回每个分类的在原样本中的序号
    #该函数真正进行分类
    def sample_classify(self,X,Y):
        class_index=[]
        cbsample=np.hstack((X,Y))
        class_feature=np.array(self.feature_list)
        #dis保存了样本到不同分类（中心点）的距离
        dis=EuclideanDistance(cbsample,class_feature)
        #0表示id=1的类，sample_class保存了每个样本的类别id-1
        sample_class=np.argmin(dis, axis=1)

        for i in range(self.class_id-1):
            index_i=np.array(np.where(sample_class==i))  #在三类中，距离第i+1类最近的数据
            class_index.append(index_i)
        return  class_index

    def sample_classify_featurelist(self,X,Y,feature_list):
        class_index = []
        cbsample = np.hstack((X, Y))
        class_feature = np.array(feature_list)
        # dis保存了样本到不同分类的距离
        dis = EuclideanDistance(cbsample, class_feature)
        # 0表示id=1的类，sample_class保存了每个样本的类别id-1
        sample_class = np.argmin(dis, axis=1)

        for i in range(len(feature_list)):
            index_i = np.array(np.where(sample_class == i))
            class_index.append(index_i)
        return class_index


def sigmoid_function(z):
    return  1/(1 + math.exp(-z))


#net_id可以用来构建网络的时候
def classify_net_main(insize,ousize,net_id,lrate,X,Y,deep,classnum,evaluated,sc_er,model_list,optimer_list, loss_func_list,model_list_pos,model_and_feature,loss_list,indeep,lrate_max,now):
    #判断是否终止递归
    print(lrate)
    if indeep<1:
        if deep<1:
            return
    elif deep<1:
        if indeep<1:
            return
        indeep=indeep-1

        # 创建本次训练中要被训练的函数
    if model_list_pos<0:
        model_new= Classify_Net(insize, ousize,net_id)
        net_id=net_id+1
        model_list_pos=0
        model_list.append(model_new)
        #这里修改model为model—list
        optimer_list.append(torch.optim.SGD(model_list[model_list_pos].parameters(), lr=lrate))
        loss_func_list.append(nn.SmoothL1Loss())
    else:
        model_new = Classify_Net(insize, ousize,net_id)
        net_id = net_id + 1
        #复制训练的参数
        #model_list_pos是要被复制的模型的位置，进行更新之后是当前模型的位置
        net_dict = model_list[model_list_pos].state_dict()  # net是已经通过net.load_state_dict加载过参数的模型
        model_dict = model_new.state_dict()  # model是跟net一样的网络，但是没有加载模型参数
        for par1, par2 in zip(net_dict, model_dict):
            model_dict[par2] = net_dict[par1]  # 这里赋值，也可以在这里修改model_dict的网络参数的值
        model_new.load_state_dict(model_dict)  ##重新load下，这里model的参数就跟net中一样了
        # 把模型添加到list
        model_list.append(model_new)
        #更新model_list_pos,使其表示现在的模型
        model_list_pos = len(model_list)-1
        optimer_list.append(torch.optim.SGD(model_list[model_list_pos].parameters(), lr=lrate))
        loss_func_list.append(nn.SmoothL1Loss())

    if len(X)>64:
        batchSize = 64
    else:
        batchSize=int(len(X)/2)
    MaxLen = len(X)
    # print('X', X)
    # print('Maxlen',MaxLen)
    # print('batchSize', batchSize)
    d_list=[]
    for i in range(len(sc_er.depth_list)):
        d_list.append(sc_er.depth_list[i])
    d_list.append(deep)
    lr_list = []  # 存储每代的学习率
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimer_list[model_list_pos], mode='min', factor=0.1, patience=1000,
    #                                                        verbose=False,threshold=0.000001, threshold_mode='rel', cooldown=0,
    #                                                        min_lr=0,eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimer_list[model_list_pos], mode='min', factor=0.5, patience=3500,
    #                                                        verbose=False,
    #                                                        threshold=0.000001, threshold_mode='rel', cooldown=0,
    #                                                        min_lr=0,
    #                                                        eps=lrate_max)
    #训练网络
    losstemp=[]
    for epoch in range(evaluated):

        seed = torch.randint(0, MaxLen - 1, (1, batchSize))
        x = X[seed]
        y = Y[seed]
        # print(y)
        x = torch.tensor(x).float().squeeze(0)
        y = torch.tensor(y).float()
        out = model_list[model_list_pos](x)

        # print('out_size:',out.size())
        loss = loss_func_list[model_list_pos](out, y)  # 计算误差
        # print('Loss',(out-y).detach().numpy())

        optimer_list[model_list_pos].zero_grad()
        loss.backward()
        optimer_list[model_list_pos].step()
        # scheduler.step(loss)
        # lr_list.append(optimer_list[model_list_pos].state_dict()['param_groups'][0]['lr'])  #####
        losstemp.append(loss)
        # print(model_list_pos)
        model_par = model_list[model_list_pos].state_dict()
        model_bais = model_par['Base.weight']
        t2 = model_bais

        #print('t2', t2.detach().numpy())
        if epoch % 1000 == 0:
            print('epoch{0} depth_list{1}'.format(epoch,d_list))
            print('wucha:',np.sum(np.sum(np.abs((out - y).detach().numpy()),axis=2)[0])/batchSize)
            print('depth{0}'.format(deep))
            print('loss',loss)
            print('t2', t2[0].detach().numpy())
            # print('current_lrate', lr_list[len(lr_list) - 1])

    losstemp_new = []
    for i in range(len(losstemp)):
        losstemp_new.append(losstemp[i].item())
    plt.plot(range(len(losstemp_new)), losstemp_new, color='r')
    now2 = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    filename = './exp_fig/method_' + now + 'loss_' + now2 + 'depth__{0}__'.format(d_list[len(d_list) - 1])
    plt.savefig(filename + '.png', dpi=200, bbox_inches='tight')
    plt.pause(1)
    plt.close()

    x = X
    y = Y
    #前后是否进行归一化要保持一致，且要在运行函数之前进行归一化而不是运行该函数之时
    #x = normalization(x)
    All_x = torch.tensor(x).float().squeeze(0)
    All_y = torch.tensor(y).float()
    All_out = model_list[model_list_pos](All_x)
    # print('All_out:',All_out)
    # print('All_Y',All_y)
    loss_for_node = np.abs((All_out - All_y).detach().numpy())
    # print('len for node',len(loss_for_node))
    # print('lossfor_node', loss_for_node)
    loss_for_node = np.sum(loss_for_node, axis=1)/ ousize
    # print('len for node', len(loss_for_node))
    # print('lossfor_node', loss_for_node)
   # print('loss',np.abs(((All_out - All_y).detach().numpy()), axis=2)[0])
    #平均误差
    loss_for_node = np.resize(loss_for_node, (np.shape(loss_for_node)[0], 1))
    # print('len for node', len(loss_for_node))
    # print('lossfor_node',loss_for_node)
    sc_er.class_feature(All_x.detach().numpy(), All_y.detach().numpy(), loss_for_node, classnum, insize, ousize, model_list[model_list_pos],
                        model_list[model_list_pos].id, deep,method='percent')

    loss_list.append(loss_for_node)
    fealen=len(sc_er.feature_list)
    feature_for_model=sc_er.feature_list[fealen-classnum:fealen]
    model_and_feature.append([model_list[model_list_pos],feature_for_model,sum(All_x)/len(All_x),len(All_x)]) #保存当前模型（结点）以下三个子模型（结点）的feature_list
    class_index = sc_er.sample_classify_featurelist(All_x, All_y,feature_for_model)
    print('class1 size',len(class_index[0][0]))
    print('class2 size',len(class_index[1][0]))
    print('class3 size', len(class_index[2][0]))
    # lrate=lr_list[len(lr_list) - 1]
    for i in range(len(class_index)):#对三组分类数据的index进行
        if len(class_index[i][0]) <= 2000:  #该组几乎没有分配到数据
            pass
        else:
            print('aver_nodeloss',np.average(loss_for_node))
            print('indeep',indeep)
            #如果误差大于2000而且deep小于1的话说明已经到设置的深度，但是
            if np.average(loss_for_node)>1000 and deep<=1:
                classify_net_main(insize, ousize, net_id, lrate, X[class_index[i][0]],
                                  Y[class_index[i][0]], deep - 1, classnum, int(evaluated ),
                                  sc_er, model_list,
                                  optimer_list, loss_func_list, model_list_pos, model_and_feature, loss_list, indeep,lrate_max,now)
                #如果误差小于2000的话不用继续分支，indeep设置为0
            elif np.average(loss_for_node)<1000:
                classify_net_main(insize, ousize, net_id, lrate, X[class_index[i][0]],
                                  Y[class_index[i][0]], deep - 1, classnum, evaluated,
                                  sc_er, model_list,
                                  optimer_list, loss_func_list, model_list_pos, model_and_feature, loss_list, 0,lrate_max,now)
            else: #等于2000或大于2000但deep还>1
                classify_net_main(insize, ousize, net_id, lrate, X[class_index[i][0]],
                                  Y[class_index[i][0]], deep - 1, classnum, int(evaluated),
                                  sc_er, model_list,
                                  optimer_list, loss_func_list, model_list_pos, model_and_feature, loss_list, indeep,lrate_max,now)



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

class Net(torch.nn.Module):
    def __init__(self,n_feature, n_output,id=-1):
        super(Net, self).__init__()
        self.id = id;
        self.Base=torch.nn.Linear(n_feature, 2)
        self.Base2 = torch.nn.Linear(2, 2)
        self.predict=torch.nn.Linear(2,n_output)
    def forward(self,x):
        x=self.Base(x)
        # x=F.relu(self.Base1(x))
        x=self.Base2(x)
        x=self.predict(x)
        return x
    def set_id(self,id):
        if self.id==-1:
            self.id=id
        else:
            print('id 已经被部署，不可重复修改')
    def reset(self):
        self.Base.parameters()
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


class Classify_Net(torch.nn.Module):
    def __init__(self, insize,outsize,id=-1):
        super(Classify_Net, self).__init__()
        self.id=id;
        self.Base=torch.nn.Linear(insize,16)
        self.Base1=torch.nn.Linear(16,32)
        # self.Base2=torch.nn.Linear(32,64)
        # self.Base31 = torch.nn.Linear(64, 128)
        # self.Base32 = torch.nn.Linear(128, 64)
        # self.Base33= torch.nn.Linear(64, 32)
        self.Base4 = torch.nn.Linear(32, 16)
        self.Out=torch.nn.Linear(16,outsize)
        self.act=torch.nn.Tanh()

    def set_id(self,id):
        if self.id==-1:
            self.id=id
        else:
            print('id 已经被部署，不可重复修改')

    def forward(self, x):
        x=F.relu(self.Base(x))
        x=F.relu(self.Base1(x))
        # x=F.relu(self.Base2(x))
        # x=F.relu(self.Base31(x))
        # x=F.relu(self.Base32(x))
        # x=F.relu(self.Base33(x))
        x=F.relu(self.Base4(x))
        x=self.Out(x)
        return  x

    def reset(self):
        self.Base.parameters()

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}



def calcul_BP_learningrate_new(Input, Output,lrate,lrate_max,evaluated):

    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])
    model_and_feature = []

    batchSize = 64
    runsize = 1  # 重复实验的次数
    groupsize = 1  # 一组里面有多少个个体,每个个体的for循环次数不同
    # evaluated = 50000  # 每组每轮训练迭代次数 暂用500
    #lrate = 0.000000095  # 学习率
    Allloss = []
    # net_id给每一个模型一个id，该id和
    net_id = 0
    losslist = []
    losstemp = []
    loss_for_node = np.array([])
    model = []
    optimer = []
    loss_func = []
    X_evaluate = []
    Y_evaluate = []
    sc_er = sample_classify()  # sample_classifier 类
    class_position_list = []
    lr_list = []  # 存储每代的学习率
    # lrate = 1e-06  # 定义初始学习率
    for i in range(groupsize):
        #初始化模型
        model.append(Classify_Net(insize, ousize, net_id))  # 生成一个group中的模型,并赋予id
        model_par = model[i].state_dict()
        model_bais = model_par['Base.weight']
        t1 = model_bais.detach().numpy()
        print("t1", t1)  #模型权重

        net_id = net_id + 1
        optimer.append(torch.optim.SGD(model[i].parameters(), lr=lrate))  # 可迭代字典

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimer[i], mode='min', factor=0.5, patience=10000, verbose=False,
        #                                            threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                            eps=lrate_max)

        losstemp=[]  # 记录在一次实验（run）中每个模型在每一次迭代的损失
        loss_func.append(nn.SmoothL1Loss())

        plt.ion()
        plt.show()

        for epoch in range(evaluated):
            optimer[i].zero_grad()
            # 选择数据并进行处理，每一次迭代选择之后的数据都应该被保存起来
            seed = torch.randint(0, MaxLen - 1, (1, batchSize))
            x = Input[seed]
            y = Output[seed]

            x = torch.tensor(x).float().squeeze(0)
            y = torch.tensor(y).float()
            out = model[i](x)
            loss = loss_func[i](out, y)  # 计算误差

            loss.backward()
            optimer[i].step()
            # scheduler.step(loss) ##### 第二类

            # lr_list.append(optimer[i].state_dict()['param_groups'][0]['lr'])  #####

            # loss_numpy = np.abs((out - y).detach().numpy()[0])
            # loss_numpy = np.sum(loss_numpy, axis=1) / np.shape(loss_numpy)[1]
            # loss_numpy = np.resize(loss_numpy, (batchSize, 1))

            losstemp.append(loss)

            model_par = model[i].state_dict()
            model_bais = model_par['Base.weight']
            t2 = model_bais
            if epoch % 100000 == 0:
                print('groups={0}, epoch{1}'.format(i + 1, epoch))
                print('loss',loss)
                # print('current_lrate',lr_list[len(lr_list)-1])
                print('t2', t2[0].detach().numpy())

            if (epoch+1)%100000==0:
                losstemp_new = []
                x_range = []
                for j in range(len(losstemp)):
                    x_range.append(j)
                    losstemp_new.append(losstemp[j].item())
                plt.cla()
                plt.plot(x_range[1:len(losstemp_new):10000], losstemp_new[1:len(losstemp_new):10000], color='r')
                plt.pause(2)
                plt.close()



    return model[i]


def calcul_BP(Input, Output,lrate,lrate_max):

    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])

    batchSize = 16
    runsize = 1  # 重复实验的次数
    groupsize = 1  # 一组里面有多少个个体,每个个体的for循环次数不同
    evaluated = 25000  # 每组每轮训练迭代次数 暂用500
    Allloss = []
    net_id = 0
    losstemp = []
    loss_for_node = np.array([])
    model = []
    optimer = []
    loss_func = []
    lr_list = []  # 存储每代的学习率

    for i in range(groupsize):
        #初始化模型
        # model.append(Net(insize, 10, ousize))  # 生成一个group中的模型,并赋予id
        model.append(Classify_Net(insize, ousize, net_id))  # 生成一个group中的模型,并赋予id
        print(model[i])
        # model_par = model[i].state_dict()
        # model_bais = model_par['Base.weight']
        # t1 = model_bais.detach().numpy()
        # print("t1", t1)  #模型权重
        net_id = net_id + 1
        optimer.append(torch.optim.SGD(model[i].parameters(), lr=lrate))  # 可迭代字典
        print(optimer[i])
        losstemp.append([])  # 记录在一次实验（run）中每个模型在每一次迭代的损失
        loss_func.append(nn.SmoothL1Loss())
        print(loss_func[i])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimer[i], mode='min', factor=0.5, patience=10000, verbose=False,
                                                   threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                   eps=lrate_max)
        for epoch in range(evaluated):

            # 选择数据并进行处理，每一次迭代选择之后的数据都应该被保存起来
            seed = torch.randint(0, MaxLen - 1, (1, batchSize))
            x = Input[seed]
            y = Output[seed]

            x = torch.tensor(x).float().squeeze(0)
            y = torch.tensor(y).float()
            out = model[i](x)
            loss = loss_func[i](out, y)  # 计算误差

            optimer[i].zero_grad()
            loss.backward()
            optimer[i].step()

            scheduler.step(loss) ##### 第二类
            lr_list.append(optimer[i].state_dict()['param_groups'][0]['lr'])  #####

            loss_numpy = np.abs((out - y).detach().numpy()[0])
            loss_numpy = np.sum(loss_numpy, axis=1) / np.shape(loss_numpy)[1]
            loss_numpy = np.resize(loss_numpy, (batchSize, 1))
            #
            losstemp[i].append(loss_numpy)

            model_par = model[i].state_dict()
            model_bais = model_par['Base.weight']
            t2 = model_bais

            if epoch % 1024 == 0:
                print('groups={0}, epoch{1}'.format(i + 1, epoch))
                print('loss',loss)
                print('current_lrate',lr_list[len(lr_list)-1])
                print('t2', t2[0].detach().numpy())

    return model[i]




def testing(Input, Output,lrate,lrate_max):
    # Output[1:10000]=1
    # Output[10001:len(Output)]=0

    Input=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    Output=Input.pow(2)+0.2*torch.rand(Input.size())

    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])

    batchSize = 16
    runsize = 1  # 重复实验的次数
    groupsize = 1  # 一组里面有多少个个体,每个个体的for循环次数不同
    evaluated = 1000  # 每组每轮训练迭代次数 暂用500
    Allloss = []
    net_id = 0
    losstemp = []
    loss_for_node = np.array([])
    model = []
    optimer = []
    loss_func = []
    lr_list = []  # 存储每代的学习率

    for i in range(groupsize):
        #初始化模型
        # model.append(Net(insize, 10, ousize))  # 生成一个group中的模型,并赋予id
        model.append(Classify_Net(1, 1, net_id))  # 生成一个group中的模型,并赋予id
        print(model[i])
        net_id = net_id + 1
        optimer.append(torch.optim.SGD(model[i].parameters(), lr=0.5))  # 可迭代字典
        print(optimer[i])
        losstemp.append([])  # 记录在一次实验（run）中每个模型在每一次迭代的损失
        loss_func.append(nn.SmoothL1Loss())
        print(loss_func[i])

        plt.ion()
        plt.show()

        for epoch in range(evaluated):


            out = model[i](Input)
            loss = loss_func[i](out, Output)  # 计算误差

            optimer[i].zero_grad()
            loss.backward()
            optimer[i].step()

            if (epoch+1)%10==0:
                plt.cla()
                plt.scatter(Input.data.numpy(),Output.data.numpy())
                plt.plot(Input.data.numpy(),out.data.numpy())
                # plt.text(0.5,0,'L=%.4f'% loss.data[0],fontdict={'size':20,'color':'red'})
                plt.pause(0.1)


            # model_par = model[i].state_dict()
            # model_bais = model_par['hidden.weight']
            # t2 = model_bais
            # print('t2', t2[0].detach().numpy())
            if epoch % 100 == 0:
                print('groups={0}, epoch{1}'.format(i + 1, epoch))
                print('loss',loss)
            #     print('current_lrate',lr_list[len(lr_list)-1])
            #     print('t2', t2[0].detach().numpy())


    return model[i]



def testing_BP(Input, Output,lrate,lrate_max):
    # Output[1:10000]=1
    # Output[10001:len(Output)]=0

    Input = torch.tensor(Input).float()
    Output = torch.tensor(Output).float()

    MaxLen = len(Input)
    insize = len(Input[1,:])
    ousize = len(Output[1,:])

    batchSize = 16
    runsize = 1  # 重复实验的次数
    groupsize = 1  # 一组里面有多少个个体,每个个体的for循环次数不同
    evaluated = 1500000  # 每组每轮训练迭代次数 暂用500
    Allloss = []
    net_id = 0
    losstemp = []
    loss_for_node = np.array([])
    model = []
    optimer = []
    loss_func = []
    lr_list = []  # 存储每代的学习率

    for i in range(groupsize):
        #初始化模型
        # model.append(Net(insize, 10, ousize))  # 生成一个group中的模型,并赋予id
        model.append(Classify_Net(insize, ousize, net_id))  # 生成一个group中的模型,并赋予id
        print(model[i])
        net_id = net_id + 1
        optimer.append(torch.optim.SGD(model[i].parameters(), lr=lrate))  # 可迭代字典
        print(optimer[i])
        losstemp=[] # 记录在一次实验（run）中每个模型在每一次迭代的损失
        loss_func.append(nn.SmoothL1Loss())
        print(loss_func[i])

        for epoch in range(evaluated):
            optimer[i].zero_grad()

            out = model[i](Input)
            loss = loss_func[i](out, Output)  # 计算误差

            # loss_numpy = np.abs((out - Output).detach().numpy()[0])
            # loss_numpy = np.sum(loss_numpy, axis=1) / np.shape(loss_numpy)[1]
            # loss_numpy = np.resize(loss_numpy, (batchSize, 1))
            losstemp.append(loss)

            loss.backward()
            optimer[i].step()

            model_par = model[i].state_dict()
            model_bais = model_par['Base.weight']
            t2 = model_bais
            # print('t2', t2[0].detach().numpy())
            if epoch % 1000== 0:
                print('groups={0}, epoch{1}'.format(i + 1, epoch))
                print('loss',loss)
            #     print('current_lrate',lr_list[len(lr_list)-1])
                print('t2', t2[0].detach().numpy())

        # plt.plot(range(evaluated), lr_list, color='r')    # 绘制出学习率曲线 #############
        # plt.show()###################

        # losstemp_new=[]
        # x_range=[]
        # for j in range(len(losstemp)):
        #     losstemp_new.append(losstemp[j].item())
        #     x_range.append(j)
        # plt.plot(x_range[1:len(losstemp_new):1000], losstemp_new[1:len(losstemp_new):1000], color='r')

    return model[i]