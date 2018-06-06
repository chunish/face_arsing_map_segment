#!/usr/bin/python
# -*- coding: utf8 -*-
import os
import sys
import argparse
from PIL import Image
from PIL import ImageDraw
import numpy as np
from torchvision import utils
from torch.utils.data import DataLoader
from torch import autograd, nn
import torch
import time
import torch.nn.functional as F
from torchvision import transforms
from loader2 import dLoader
from loss_def import CrossEntropyLoss2d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test import Test
import Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, nargs='+', default='/home/yiqi.liu-2/hanchun.shen/PM_Net/data/Img_paths.txt')
    parser.add_argument('--labels_dir', type=str, default='/home/yiqi.liu-2/hanchun.shen/PM_Net/data/labels_dir/')
    parser.add_argument('--labels_path', type=str, default='/home/yiqi.liu-2/hanchun.shen/PM_Net/data/labels/')
    parser.add_argument('--save_path', type=str, default='/home/yiqi.liu-2/hanchun.shen/PM_Net/data/results/')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training. Total for all gpus.")
    parser.add_argument('--momentum', type=float, default=0.99, help="momentum")
    parser.add_argument('--weight_decay', type=int, default=2e-5, help="weight_decay")
    parser.add_argument('--max_epoch', type=int, default=1000, help="Training will be stoped in this case.")
    parser.add_argument('--img_size', type=int, default=128, help="The size of input for model")
    parser.add_argument('--process_num', type=int, default=20, help="The number of process to preprocess image.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="lr")
    parser.add_argument('--gpu_device', type=str, default='0,1,2,3', help="GPU index. Support Multi GPU. eg: 1,2,3")
    parser.add_argument('--img_format', type=str, default='RGB', help="The color format for training.")
    return parser.parse_args()


def train():
	args = get_args()
	# tst = Test()
	# load data
	transformations = transforms.Compose([transforms.ToTensor()])

	datas =  dLoader(args.img_size, args.input_path, args.labels_dir, args.labels_path, transformations)

	data_loader = DataLoader(datas, batch_size = args.batch_size, 
		shuffle = True, num_workers = 0, drop_last = True)
	print('---------------------Total data LOADED!---------------------\n')
	model = Model.PM_Net()
	if torch.cuda.is_available():
		model = model.cuda()
		model.load_state_dict(torch.load(args.save_path + 'PM_Net_' + str(25) + 'E_checkpoint.pkl'))

	# loss function
	#criterion = nn.L1Loss(size_average = False).cuda()
	criterion = CrossEntropyLoss2d().cuda()
	optimizor = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)#lr = args.learning_rate)

	print('---------------------Start TRAINING!---------------------\n')
	fw = open(args.save_path + 'loss.txt','a')
	# tst = Test()
	ts = time.time()
	tt = time.time()
	for epoch in range(args.max_epoch):
		y = []
		for it, data in enumerate(data_loader):
			#========== farward
			train_input, mask = data
			train_input = autograd.Variable(train_input.cuda())
			mask = autograd.Variable(mask.cuda())
			
			optimizor.zero_grad()
			
			train_out = model(train_input)  # [16, 11, 128, 128]
			loss = criterion(train_out, mask)
			
			#========== Backward
			loss.backward()
			optimizor.step()
			if it % 10 is 0:
				print('Training Phase: [%2d][%2d/%2d]\tLoss: %.5f  Time: [%2.3f/%6.3f]' %
				  (it, epoch, args.max_epoch, loss.data[0], time.time() - tt, time.time() - ts))
				tt = time.time()
			y.append(loss.data[0])
			fw.write(str(loss.data[0]) + '\n')
			# tst.run((train_out.data).cpu().numpy())
			it += 1



		print 'Epoch {} finished! Time cost: {}s'.format(epoch, time.time() - ts)
		if (1 + epoch) % 25 == 0:
			torch.save(model.state_dict(),args.save_path + 'PM_Net_' + str(epoch + 1) + 'E_checkpoint.pkl')
				#print(train_outputs.shape)

if __name__ == "__main__":
    train()