from __future__ import print_function

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Stack(nn.Module):

	"""
	Neural stack implementation based on Grefensette et al., 2015.
	@see https://arxiv.org/pdf/1506.02516.pdf
	"""

	def __init__(self, batch_size, embedding_size):
		super(Stack, self).__init__()

		# initialize tensors
		self.V = Variable(torch.FloatTensor(0))
		self.s = Variable(torch.FloatTensor(0))

		# TODO either make everything a variable
		# TODO or implement custom backward pass function
		# I chose the first approach :)

		self.zero = Variable(torch.zeros(batch_size))

		self.batch_size = batch_size
		self.embedding_size = embedding_size
	
	def read(self):
		"""
		it's just like pop(1)!
		"""
		ones = Variable(torch.ones(self.batch_size), requires_grad=False)
		res= self.pop(ones)

		weights = self.s - res # [t, batch_size]
		weights.unsqueeze_(-1)
		t = weights.expand(*self.V.size()) * self.V
		r = torch.sum(t, 0) # [batch_size, embedding_size]

		return r

	def pop(self, w):
		s = Variable(torch.FloatTensor(self.s.data))
		# note: we do this bc pytorch doesn't support reverse slice
		# if s has NO dimensions, s.size(0) won't work
		if len(s.size()) == 0:
			return s		
		
		idx = [i for i in range(s.size(0)-1, -1, -1)] 

		top = s[idx]
		top = torch.cumsum(top, 0) - w # this is what pop looks like
		top = top.clamp(min=0)

		# to get out of summed weights
		if len(idx) > 1:
			# print idx
			reverse_sum = torch.cat((torch.zeros_like(top[:1]), top[:-1]), 0)
			top -= reverse_sum 		
		s = top[idx]

		return s

	# TODO initialize stack to fixed size

	def forward(self, v, u, d):
		"""
		@param v [batch_size, embedding_size] matrix to push
		@param u [batch_size,] vector of pop signals in (0, 1)
		@param d [batch_size,] vector of push signals in (0, 1)
		@return [batch_size, embedding_size] read matrix
		"""

		# update self.V
		v = v.view(1, self.batch_size, self.embedding_size)
		self.V = torch.cat([self.V, v], 0) if len(self.V.data) != 0 else v

		# If we create a new variable every time it goes forward, this means that we're not learning about what u should be ... I don't get this
		w = Variable(torch.FloatTensor(u.data), requires_grad=False)

		s = self.pop(w) 		# [t, batch_size]
		if len(s.data) == 0:
			if len(d.size()) == 1:
				dummy = Variable(d.data.unsqueeze_(-1))
			else:
				dummy = d
			self.s  = dummy
		else:
			self.s  = torch.cat((s, d), 0) 	# [t+1, batch_size]

		r = self.read()
		# r.register_hook(print)
		return r

	def log(self):
		"""
		Prints a representation of the stack to stdout.
		"""
		V = self.V.data
		if not V.shape:
			print("[Empty stack]")
			return
		for b in xrange(self.batch_size):
			if b > 0:
				print("----------------------------")
			for i in xrange(V.shape[0]):
				print("{:.2}\t|\t{:.2f}".format("\t".join(str(x) for x in V[i, b,:]), self.s[i, b].data[0]))

if __name__ == "__main__":
	print("Running stack tests..")
	stack = Stack(1, 1)
	stack.log()
	out = stack.forward(
		Variable(torch.FloatTensor([[1]])),
		Variable(torch.FloatTensor([[0]])),
		Variable(torch.FloatTensor([[.8]])),
	)
	print("\n\n")
	stack.log()
	print("read", out)
	out = stack.forward(
		Variable(torch.FloatTensor([[2]])),
		Variable(torch.FloatTensor([[.1]])),
		Variable(torch.FloatTensor([[.5]])),
	)
	print("\n\n")
	stack.log()
	print("read", out)
	out = stack.forward(
		Variable(torch.FloatTensor([[3]])),
		Variable(torch.FloatTensor([[.9]])),
		Variable(torch.FloatTensor([[.9]])),
	)
	print( "\n\n")
	stack.log()
	print ("read", out)

	""" Expected Output:

Running stack tests..
[Empty stack]



1.      |       0.80
read Variable containing:
 0.8000
[torch.FloatTensor of size 1x1]




1.      |       0.70
2.      |       0.50
read Variable containing:
 1.5000
[torch.FloatTensor of size 1x1]




1.      |       0.30
2.      |       0.00
3.      |       0.90
read Variable containing:
 2.8000
[torch.FloatTensor of size 1x1]
"""
