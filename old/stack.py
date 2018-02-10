import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Stack(nn.Module):

	def __init__(self, batch_size, embedding_size):
		super(Stack, self).__init__()

	def forward(self, v, u, d):
		"""
		@param v [batch_size, embedding_size] matrix to push
		@param u [batch_size,] vector of pop signals (0,1)
		@param d [batch_size,] vector of push signals (0,1)
		@return [batch_size, embedding_size] thing you readha
		"""
		# what happens if you pop an empty stack again?

		


