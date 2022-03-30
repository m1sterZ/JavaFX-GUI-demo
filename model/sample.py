#BPNN
#3
#ReLU Sigmoid none
#SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, neserov=False)
#BCELoss(weight=None, size_average=True, reduce=True, reduction='mean')
#1000
#576 120 84
#120 60 8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BPNN(nn.Module): 
	def __init__(self):
		super(BPNN, self).__init__()
		self.fc1 = nn.Linear(576, 120)
		self.fc2 = nn.Linear(120, 60)
		self.fc3 = nn.Linear(84, 8)

	def forward(self, x): 
		x = F.ReLU(self.fc1(x))
		x = F.Sigmoid(self.fc2(x))
		x = self.fc3(x)
		return x

net = BPNN()
criterion = nn.BCELoss(weight=None, size_average=True, reduce=True, reduction='mean')
optimizer = optim.SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, neserov=False)
for epoch in range(1000):
	running_loss = 0.0
	correct = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		predicted = torch.max(outputs.data, 1)[1]
		correct += (predicted == labels).sum()
# print running loss
	if epoch % 200 == 199:
		print('epoch:%d, loss:%.3f % (epoch + 1, running_loss / 200)')
		print('correct:%.3f % (correct / 200)')
		running_loss = 0.0
		correct = 0.0
