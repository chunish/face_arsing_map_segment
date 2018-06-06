#!/usr/bin/python
import os
import sys
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image


def FindErrPots(array):
    ashape = array.shape   
    out2 = []
    out0 = []
    for i in range(ashape[0]):
        for j in range(ashape[1]):
            if 2 == array[i][j]:
                out2.append([i, j])
                #icout2 += 1
            elif 0 == array[i][j]:
                out0.append([i, j])
                #icout0 += 1
    return np.array(out0), np.array(out2)

def redefPixel2(d_coord, labels):
    for i in d_coord:
        ic = 0
        for j in range(11):
            if labels[j,i[0],i[1]] and 0 == ic:
                labels[j,i[0],i[1]] = True
                ic += 1
            else:
                labels[j,i[0],i[1]] = False
    return labels

def redefPixl0(z_coord, labels): #sumlabels
    for i in z_coord:
            if i[0] < 127 and labels[i[0] + 1][i[1]] != 0:
                labels[i[0]][i[1]] = labels[i[0] + 1][i[1]]
            else:
                if 0 != labels[i[0] - 1][i[1]]:
                    labels[i[0]][i[1]] = labels[i[0] - 1][i[1]]
                else:
                    labels[i[0]][i[1]] = 1
    return labels

def loadLabels(ldir, idir):
    out = []
    for i, it in enumerate(ldir):
        img = Image.open(idir + it.strip('\r\n'))
        img = img.convert('1')
        out.append(np.array(img.resize((128, 128))))
    return np.array(out)

def Tomask(ldir, idir):#[11, 16, 16]
    #summ = labels.sum(axis = 0)
    labels = loadLabels(ldir, idir)
    iniSum = labels.sum(axis = 0)
    z_coord, a_coord = FindErrPots(iniSum)
    #print('------------------------------------',labels.shape)
    dst_label = redefPixel2(a_coord, labels)
    reSum = dst_label.sum(axis = 0)
    
    #print(dst_label.shape)
    summ = dst_label[1] * 1+ dst_label[2] * 2 + dst_label[3] * 3 + dst_label[4] * 4 + \
     dst_label[5] * 5 + dst_label[6] * 6 + dst_label[7] * 7 + dst_label[8] * 8 + dst_label[9] * 9 + dst_label[10] * 10
    dst = redefPixl0(z_coord, summ)
    for i in range(len(dst)):
    	for j in range(len(dst[i])):
    		if dst[i][j] > 10:
    			dst[i][j] = 10

    return np.array(dst)


class dLoader(Dataset):
    """docstring for dLoader"""
    def __init__(self, img_size, Img_txt_path, Label_dir, Label_img_path, transform = None):
        # read imgs path/file_names
        self.transform = transform
        img_paths = []
        with open(Img_txt_path) as ff:
            for line in ff:
                img_paths.append(line.strip('\r\n'))

        #read label imgs
        label_txt_path = os.listdir(Label_dir)
        full_labels = []
        for line in label_txt_path:
            with open(Label_dir + line) as ff:
                pngs = []
                for lab in ff:
                    pngs.append(lab)
                full_labels.append(pngs)

        self.img_filename = img_paths
        self.label = full_labels
        self.labeli = Label_img_path
        self.img_size = img_size

    def __getitem__(self, index):
        img = Image.open(self.img_filename[index])
        img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size)) 
        
        # Read Labels
        mask = np.zeros((self.img_size, self.img_size))
        mask = Tomask(self.label[index], self.labeli) 

        #with open('{}_.txt'.format(index), 'w') as fw:
        #    for i in mask:
        #        for j in i:
        #            fw.write(str(j) + ' ')
        #        fw.write('\n')
        #    fw.close

        if self.transform is not None:
            img = self.transform(img)
            mask = torch.from_numpy(mask)
        return img, mask# shape of img, labels like: [h, w, c], [h, w, labels]
    def __len__(self):
        return len(self.img_filename)


