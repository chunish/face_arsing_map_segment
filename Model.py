#! /usr/bin/python
# -*- coding: utf8 -*-
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class PM_Net(nn.Module):
	"""docstring for PM_Net"""
	def __init__(self):
		super(PM_Net, self).__init__()
		self.convs1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding = 1), #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding = 1),
			nn.ReLU())
		self.pool1 = nn.MaxPool2d(2, stride=2, return_indices = True)
		self.convs2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding = 1),
			nn.ReLU())
		self.pool2 = nn.MaxPool2d(2, stride=2, return_indices = True)
		self.convs3 = nn.Sequential(
			nn.Conv2d(128, 256, 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, padding = 1),
			nn.ReLU())
		self.pool3 = nn.MaxPool2d(2, stride=2, return_indices = True)
		self.convs4 = nn.Sequential(
			nn.Conv2d(256, 512, 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding = 1),
			nn.ReLU())
		self.pool4 = nn.MaxPool2d(2, stride=2, return_indices = True)
		self.convs5 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding = 1),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding = 1),
			nn.ReLU())
		self.pool5 = nn.MaxPool2d(2, stride=2, return_indices = True)


		self.fc_conv = nn.Sequential(
			nn.Conv2d(512, 4096, 1),
			nn.ReLU(),
			nn.Dropout(p = 0.5),
			nn.Conv2d(4096, 512, 1),
			nn.ReLU(),
			nn.Dropout(p=0.5))



		self.unpool1 = nn.MaxUnpool2d((2, 2), stride = (2, 2))
		self.deconvs1 = nn.Sequential(
			nn.Conv2d(512, 512, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(512, 512, 5, padding = 2),
			nn.ReLU(),
			nn.Dropout(p=0.5))
			#nn.Dropout(p = 0.5),
		self.unpool2 = nn.MaxUnpool2d((2, 2), (2, 2))
			
		self.deconvs2 = nn.Sequential(
			#nn.Upsample(scale_factor=2),,
			nn.Conv2d(512, 256, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(256, 256, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(256, 256, 5, padding = 2),
			nn.ReLU(),
			nn.Dropout(p=0.5))

			#nn.Dropout(p = 0.5),
		self.unpool3 = nn.MaxUnpool2d((2, 2), (2, 2))
		self.deconvs3 = nn.Sequential(
			nn.Conv2d(256, 128, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(128, 128, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(128, 128, 5, padding = 2),
			nn.ReLU(),
			nn.Dropout(p=0.5))

			#nn.Dropout(p = 0.5),
		self.unpool4 = nn.MaxUnpool2d((2, 2), (2, 2))
		self.deconvs4 = nn.Sequential(
			nn.Conv2d(128, 64, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(64, 64, 5, padding = 2),
			nn.ReLU(),
			nn.Dropout(p=0.5))

			#nn.Dropout(p = 0.5),
		self.unpool5 = nn.MaxUnpool2d((2, 2), (2, 2))
		self.deconvs5 = nn.Sequential(
			nn.Conv2d(64, 32, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(32, 32, 5, padding = 2),
			nn.ReLU(),
			#nn.Dropout(p = 0.5),
			nn.Conv2d(32, 11, 3, padding = 1)
			)
		
	def forward(self, x):
		#print('---------------------Start forward!\n')
		x = self.convs1(x)
		x, ind1 = self.pool1(x)
		x = self.convs2(x)
		x, ind2 = self.pool2(x)
		x = self.convs3(x)
		x, ind3 = self.pool3(x)
		x = self.convs4(x)
		x, ind4 = self.pool4(x)		
		x = self.convs5(x)
		x, ind5 = self.pool5(x)
		x = self.fc_conv(x)
		x = self.unpool1(x, ind5)
		x = self.deconvs1(x)
		x = self.unpool2(x, ind4)
		x = self.deconvs2(x)
		x = self.unpool3(x, ind3)
		x = self.deconvs3(x)
		x = self.unpool4(x, ind2)
		x = self.deconvs4(x)
		x = self.unpool5(x, ind1)
		x = self.deconvs5(x)
		return x




