# -*- coding: utf8 -*-
import torch
import datetime
import os
import time
import numpy as np
from PIL import Image
from torch import autograd, nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from Model import PM_Net
from loader2 import dLoader

class Test(object):
    def __init__(self):
        self.model_path = '/home/yiqi.liu-2/hanchun.shen/PM_Net/data/results/PM_Net_250D_checkpoint.pkl'
        self.__main_dir = '/home/yiqi.liu-2/hanchun.shen/PM_Net/data/test/dsts/'
        self.batch_size = 2
        self.__img_size = 128
        self.__loadPath = '/home/yiqi.liu-2/hanchun.shen/PM_Net/data/test/srcs/'
        self.__color = [[255, 255, 255],
                        [0, 175, 131],
                        [0, 111, 157],
                        [208, 221, 50],
                        [244, 157, 37],
                        [101, 31, 79],
                        [0, 21, 40],
                        [49, 89, 97],
                        [105, 51, 29],
                        [19, 194, 255],
                        [230, 49, 101]]

    def __save_path(self):

        tt = datetime.datetime.now().strftime('%d%H%M%S%f') + '_' + str(np.random.random_integers(0, 1000)).zfill(3)
        return self.__main_dir + tt + '.png'

    def __label2img(self, label):
        img = np.zeros((label.shape[0], label.shape[1], 3))
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                img[i, j, :] = self.__color[label[i, j]]
        return np.uint8(img)

    def __load(self):
        paths = [self.__loadPath + pi for pi in os.listdir(self.__loadPath)]

        transf = transforms.Compose([transforms.ToTensor()])
        datas = dLoader(self.__img_size, paths,
                        '', '', transf)

        data_loader = DataLoader(datas, batch_size=self.batch_size,
                                 shuffle=False, num_workers=0, drop_last=True)
        return data_loader

    def fake(self):
        a = np.random.random_sample((16, 11, 128, 128))
        return a

    def run(self):
        print '--------TESTING-----------'
        ts = time.time()
        src_data = self.__load()
        print 'Inputs loaded: %3fs' % (time.time() - ts)
        ts = time.time()
        net = PM_Net()
        net.eval()
        print 'Net loaded: %3fs' % (time.time() - ts)
        ts = time.time()
        if torch.cuda.is_available():
            net = net.cuda()
        net.load_state_dict(torch.load(self.model_path))
        print 'Model loaded: %3fs' % (time.time() - ts)
        tt = time.time()


        for it, data in enumerate(src_data):  # every batch
            train_input = autograd.Variable(data.cuda())
            train_out = net(train_input)

            datain = make_grid()


            for ii, im in enumerate((train_out.data).cpu().numpy()):
                im = np.array(im)
                # print im.shape
                labelimg = im.argmax(axis=0)
                img = self.__label2img(labelimg)
                rst = Image.fromarray(img)
                rst.save(self.__save_path())
                print 'Model test: [{}/{}]\t[time: {}]'.format(ii,it, time.time() - tt)
                tt = time.time()


        # for im in src:
        #     im = np.array(im)
        #     print im.shape
        #     labelimg = im.argmax(axis=0)
        #     img = self.__label2img(labelimg)
        #     rst = Image.fromarray(img)
        #     rst.save(self.__save_path())

if __name__ == '__main__':
    t = time.time()
    test = Test()
    test.run()
    print 'Whole time: ',time.time() - t



