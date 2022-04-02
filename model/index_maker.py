import random
import numpy as np

# 新数据集来此确定index
filename= 'data_solution_new_1_1'  #包名
length=1000000   #数据集长度
testing_number = int(length * 0.03)
data_index=[i for i in range(length)]
testing_index = random.sample(data_index, testing_number)  # 测试集indexes
for i in testing_index:
    data_index.remove(i)
training_index = data_index  # 训练集indexes
a=[]
a.append(training_index)
a.append(testing_index)
np.save('./'+filename+'/data_index.npy', a)