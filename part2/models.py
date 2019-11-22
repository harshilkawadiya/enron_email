import torch
import torch.nn as nn
from torch.autograd import Variable

class Classifier(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, num_classes):
		super(Classifier,self).__init__()
		self.num_classes = num_classes
		self.num_layers = 1
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(embedding_dim,hidden_dim)
		self.fc = nn.Linear(hidden_dim,self.num_classes)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self,x):
		batch_size = 1
		state = self._init_state(b_size=batch_size)
		x_out,x_hid =  self.lstm(x.reshape(x.shape[0],batch_size,x.shape[1]),state)
		out = self.fc(x_hid[0][-1])
		# print('Out Shape : ',out.shape,out)
		return self.softmax(out)

	def _init_state(self, b_size=1):
		weight = next(self.parameters()).data
		return (
		    Variable(weight.new(self.num_layers, b_size, self.hidden_dim).normal_(0.0, 0.01)),
		    Variable(weight.new(self.num_layers, b_size, self.hidden_dim).normal_(0.0, 0.01))
		)