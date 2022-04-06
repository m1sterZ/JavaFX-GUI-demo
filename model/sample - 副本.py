#1
#BPNN
#3
#ReLU Sigmoid none
#SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, neserov=False)
#BCELoss(weight=None, size_average=True, reduce=True, reduction='mean')
#2000
#576 120 84
#120 64 8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import classifier as clf
import random

class BPNN(nn.Module): 
	def __init__(self, insize, outsize):
		super(BPNN, self).__init__()
		self.fc1 = nn.Linear(insize, 16)
		self.fc2 = nn.Linear(16, 16)
		self.fc3 = nn.Linear(16, outsize)

	def forward(self, x): 
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def train(inputs, labels):
	insize = len(inputs[1,:])
	outsize = len(labels[1,:])
	net = BPNN(insize, outsize)
	criterion = nn.SmoothL1Loss()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0)
	for epoch in range(2):
		running_loss = 0.0
		for i in range(len(inputs)):
			seed = random.randint(0, len(inputs) - 1)
			x = inputs[seed]
			y = labels[seed]
			x = torch.tensor(x).float()
			y = torch.tensor(y).float()
			optimizer.zero_grad()
			outputs = net(x)
			loss = criterion(outputs, y)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
# print running loss
			if i % 2000 == 1999:
				print('epoch:%d, loss:%.3f' % (epoch + 1, running_loss / 2000))
				running_loss = 0.0
	wrapper = clf.save_model_wrapper(net, inputs, labels)
	return wrapper
