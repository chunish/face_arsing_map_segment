#!/usr/bin/python
# -*- coding: utf8 -*-
from torch import nn
import torch.nn.functional as F
class CrossEntropyLoss2d(nn.Module):
	"""docstring for CrossEntropyLoss2d"""
	def __init__(self, weight=None, size_average=True, ignore_index=255):
		super(CrossEntropyLoss2d, self).__init__()
		self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

	def forward(self, inputs, targets):
		return self.nll_loss(F.log_softmax(inputs), targets)
