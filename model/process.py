import os
#for reading utf-8
import codecs
# # encoding utf-8

# # write
# file_path = "sample.py"
# with open(file_path, "a+") as f:
#     f.write("hello world\n")
    # print("1")

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
    strs = [None] * 8
    i = 0
    for line in codecs.open(file_path, encoding='utf-8'):
        strs[i] = line[1:-1]
        i = i + 1
    net_name = strs[0]
    level = strs[1]
    activators = strs[2].split(" ")
    optimizer = strs[3]
    loss_func = strs[4]
    epoch = strs[5]
    in_di = strs[6].split(" ")
    out_di = strs[7].split(" ")
    print(out_di)

    with open(file_path, "a+") as f:
        f.write("import torch\n")
        f.write("import torch.nn as nn\n")
        f.write("import torch.nn.functional as F\n")
        f.write("import torch.optim as optim\n")
        f.write("\nclass " + net_name + "(nn.Module): \n")
        f.write("\tdef __init__(self):\n")
        f.write("\t\tsuper(" + net_name + ", self).__init__()\n")

        for l in range(int(level)):
            n = l + 1
            f.write("\t\tself.fc" + str(n) + " = nn.Linear(" + in_di[l] + ", " + out_di[l] + ")\n")

        # forward
        f.write("\n\tdef forward(self, x): \n")
        for l in range(int(level)):
            n = l + 1
            if activators[l] != "null":
                f.write("\t\tx = F." + activators[l] + "(self.fc" + str(n) + "(x))\n")
            else:
                f.write("\t\tx = self.fc" + str(n) + "(x)\n")
        f.write("\t\treturn x\n")

        f.write("\nnet = " + net_name + "()\n")

        # training
        f.write("criterion = nn." + loss_func + "\n")
        f.write("optimizer = optim." + optimizer + "\n")
        f.write("for epoch in range(" + epoch + "):\n")
        f.write("\trunning_loss = 0.0\n")
        f.write("\tfor i, data in enumerate(trainloader, 0):\n")
        f.write("\t\tinputs, labels = data\n")
        f.write("\t\toptimizer.zero_grad()\n")
        f.write("\t\toutputs = net(inputs)\n")
        f.write("\t\tloss = criterion(outputs, labels)\n")
        f.write("\t\tloss.backward()\n")
        f.write("\t\toptimizer.step()\n")
        f.write("\t\trunning_loss += loss.item()\n")
        f.write("\t\t# print running loss\n")
        # print running loss

        
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
