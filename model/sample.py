#1
#BPNN
#3
#relu relu none
#SGD(net.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
#SmoothL1Loss(size_average=True, reduce=True, reduction='mean', beta=1.0)
#2
#10 16 32
#16 32 10
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
		self.fc2 = nn.Linear(16, 32)
		self.fc3 = nn.Linear(32, outsize)

	def forward(self, x): 
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def train(inputs, labels):
	insize = len(inputs[1,:])
	outsize = len(labels[1,:])
	net = BPNN(insize, outsize)
	criterion = nn.SmoothL1Loss(size_average=True, reduce=True, reduction='mean', beta=1.0)
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
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
			if i % 2000 == 1999:
				print('epoch:%d, loss:%.3f' % (epoch + 1, running_loss / 2000))
				running_loss = 0.0
	wrapper = clf.save_model_wrapper(net, inputs, labels)
	return wrapper
