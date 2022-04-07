import os
#for reading utf-8
import codecs
# # encoding utf-8

# # write
# file_path = "sample.py"
# with open(file_path, "a+") as f:
#     f.write("hello world\n")
    # print("1")

# 节点id
# 网络名
# 层数
# 各层激活函数
# 优化器
# 损失函数
# 迭代次数
# 各层输入维度
# 各层输出维度

def generate(file_path):
    #preprocess
    strs = [None] * 9
    i = 0
    for line in codecs.open(file_path, encoding='utf-8'):
        strs[i] = line[1:-1]
        i = i + 1
    net_name = strs[1]
    level = strs[2]
    activators = strs[3].split(" ")
    optimizer = strs[4]
    loss_func = strs[5]
    epoch = strs[6]
    in_di = strs[7].split(" ")
    out_di = strs[8].split(" ")


    with open(file_path, "a+") as f:
        f.write("import torch\n")
        f.write("import torch.nn as nn\n")
        f.write("import torch.nn.functional as F\n")
        f.write("import torch.optim as optim\n")
        f.write("import classifier as clf\n")
        f.write("import random\n")

        f.write("\nclass " + net_name + "(nn.Module): \n")
        f.write("\tdef __init__(self, insize, outsize):\n")
        f.write("\t\tsuper(" + net_name + ", self).__init__()\n")

        for l in range(int(level)):
            n = l + 1
            if n == 1:
                f.write("\t\tself.fc" + str(n) + " = nn.Linear(insize, " + out_di[l] + ")\n")
            elif n == int(level):
                f.write("\t\tself.fc" + str(n) + " = nn.Linear(" + in_di[l] + ", outsize)\n")
            else:
                f.write("\t\tself.fc" + str(n) + " = nn.Linear(" + in_di[l] + ", " + out_di[l] + ")\n")

        # forward
        f.write("\n\tdef forward(self, x): \n")
        for l in range(int(level)):
            n = l + 1
            if activators[l] != "none":
                f.write("\t\tx = F." + activators[l] + "(self.fc" + str(n) + "(x))\n")
            else:
                f.write("\t\tx = self.fc" + str(n) + "(x)\n")
        f.write("\t\treturn x\n")

        f.write("\ndef train(inputs, labels):\n")
        f.write("\tinsize = len(inputs[1,:])\n")
        f.write("\toutsize = len(labels[1,:])\n")
        f.write("\tnet = " + net_name + "(insize, outsize)\n")

        # training
        f.write("\tcriterion = nn." + loss_func + "\n")
        f.write("\toptimizer = optim." + optimizer + "\n")

        f.write("\tfor epoch in range(" + epoch + "):\n")
        f.write("\t\trunning_loss = 0.0\n")
        # f.write("\t\tcorrect = 0.0\n")

        f.write("\t\tfor i in range(len(inputs)):\n")
        f.write("\t\t\tseed = random.randint(0, len(inputs) - 1)\n")
        f.write("\t\t\tx = inputs[seed]\n")
        f.write("\t\t\ty = labels[seed]\n")
        f.write("\t\t\tx = torch.tensor(x).float()\n")
        f.write("\t\t\ty = torch.tensor(y).float()\n")
        f.write("\t\t\toptimizer.zero_grad()\n")
        f.write("\t\t\toutputs = net(x)\n")
        f.write("\t\t\tloss = criterion(outputs, y)\n")
        f.write("\t\t\tloss.backward()\n")
        f.write("\t\t\toptimizer.step()\n")
        f.write("\t\t\trunning_loss += loss.item()\n")

        # f.write("\t\t\tpredicted = torch.max(outputs.data, 1)[1]\n")
        # f.write("\t\t\tcorrect += (predicted == labels[i]).sum()\n")
        # f.write("# print running loss\n")

        f.write("\t\t\tif i % 2000 == 1999:\n")
        f.write("\t\t\t\tprint('epoch:%d, loss:%.3f' % (epoch + 1, running_loss / 2000))\n")
        # f.write("\t\t\tprint('correct:%.3f % (correct / 200)')\n")
        f.write("\t\t\t\trunning_loss = 0.0\n")
        # f.write("\t\t\tcorrect = 0.0\n")
        # print running loss

        f.write("\twrapper = clf.save_model_wrapper(net, inputs, labels)\n")
        f.write("\treturn wrapper\n")

        
    #     f.write("correct = 0.0\n")
    #     f.write("total = 0\n")
    #     f.write("with torch.no_grad():\n")
    #     f.write("\tfor data in testloader:\n")
    #     f.write("")
        # f.write("\t")

        # example: 
        # optim.SGD(params, lr=<float>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        # optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        # optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        # optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
        # optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        # optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        # optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
generate("sample.py")
