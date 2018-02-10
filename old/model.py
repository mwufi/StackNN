import torch
import torch.autograd as autograd
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

torch.manual_seed(1)

class FFController(nn.Module):
	def __init__(self, batch_size, num_embeddings, embedding_size, read_size, output_size):
		super(FFController, self).__init__()

		self.read_size = read_size
		self.batch_size = batch_size

		# controller is an embedding + a linear layer
		self.embed = nn.Embedding(num_embeddings, embedding_size)
		self.embed.weight.data.uniform_(-0.1, 0.1) # why this particular initialization?

		self.linear = nn.Linear(embedding_size + read_size, 2 + read_size + output_size)
		self.linear.weight.data.uniform_(-0.1, 0.1)
		self.linear.bias.data.fill_(0)

	def forward(self, x):
		# print(x.shape, self.read.shape)
		hidden = self.embed(x) # hidden is [embedding_size]
		output = self.linear(torch.cat([hidden, self.read], 1)) # output is [stuff, 2 + read_size + output_size]
		
		# the first part tells us how much to read
		read_params = F.sigmoid(output[:, :2+self.read_size])
		u,d,v = read_params[:,0].contiguous(), read_params[:,1].contiguous(), read_params[:,2:].contiguous()
		self.read = self.stack.forward(v.data, u.data, d.data)

		# the second part is the output
		return output[:, 2+self.read_size]

	def init_stack(self):
		self.read = torch.zeros([self.batch_size, self.read_size])
		self.stack = Stack(self.batch_size, self.read_size)

