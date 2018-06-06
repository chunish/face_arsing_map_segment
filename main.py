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

icount = 0

def arr2img(array):
	img11 = np.transpose(array, (2, 0, 1)) #[128, 128, 11]

def save_Imgs(tensor):
	#print(tensor[4])
	global icount
	labels = (tensor.data).cpu().numpy() # shape:[8,11,128,128]
	args = get_args()
	#print(labels.shape)
	print(labels[0][1])
	#imgR = transforms.ToPILImage()(labels[3]).convert('L')
	#tmp = labels[3]
	#out = np.zeros((128, 128))
	#for i in range(128):
	#	for j in range(128):
	#		if tmp[i][j] > 0.5:
	#			out[i][j] = 255
	#			print(i, j)

	#rest = Image.fromarray(np.uint8(out), 'L')
	##print(out)
	#rest.save('Yaa{}.png'.format(icount),'PNG')
	
	icount += 1

def train():
	args = get_args()
	# tst = Test()
	# load data
	print('---------------------Start loading data!---------------------\n')
	transformations = transforms.Compose([transforms.ToTensor()])

	datas =  dLoader(args.img_size, args.input_path, args.labels_dir, args.labels_path, transformations)

	data_loader = DataLoader(datas, batch_size = args.batch_size, 
		shuffle = True, num_workers = 0, drop_last = True)
	print('---------------------Finish loading data!---------------------\n')
	model = Model.PM_Net()
	if torch.cuda.is_available():
		model = model.cuda()
		# model.load_state_dict(torch.load(args.save_path + 'PM_Net_' + str(275) + 'D_checkpoint.pkl'))

	# loss function
	#criterion = nn.L1Loss(size_average = False).cuda()
	criterion = CrossEntropyLoss2d().cuda()
	optimizor = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)#lr = args.learning_rate)

	print('---------------------Start TRAINING!---------------------\n')
	fw = open(args.save_path + 'loss.txt','a')
	# tst = Test()
	tt = time.time()

	for epoch in range(args.max_epoch):
		y = []
		ts = time.time()
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
			
			print('Training Phase: Epoch: [%2d][%2d/%2d]\tIteration Loss: %.5f  Batch time: %5f' %
              (it, epoch, args.max_epoch, loss.data[0], time.time() - tt))
			tt = time.time()
			y.append(loss.data[0])
			fw.write(str(loss.data[0]) + '\n')
			# tst.run((train_out.data).cpu().numpy())
			it += 1
			# if it % 50 == 0:
				# tst.run(train_out)

		# x = np.linspace(0, 1, len(y))
		# plt.close('all')
		# plt.plot(x, y, 'r')
		# plt.savefig(args.save_path + 'epoch_{}.png'.format(epoch))
			# print(train_out.shape)
		print 'Epoch {} finished! Time cost: {}s'.format(epoch, time.time() - ts)
		if (1 + epoch) % 25 == 0:
			torch.save(model.state_dict(),args.save_path + 'PM_Net_' + str(epoch + 1) + 'E_checkpoint.pkl')
				#print(train_outputs.shape)

if __name__ == "__main__":
    train()