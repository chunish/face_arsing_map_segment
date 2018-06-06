#!/usr/bin/python
# -*- coding: utf8 -*-
import os
import sys
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

class dLoader(Dataset):
    def __init__(self, img_size, Img_txt_path, Label_dir, Label_img_path, transform = None):
        # read imgs path/file_names
        self.transform = transform
        img_paths = []
        if type(Img_txt_path) is list:
            self.img_filename = Img_txt_path
        elif type(Img_txt_path) is str:
            with open(Img_txt_path) as ff:
                for line in ff:
                    img_paths.append(line.strip('\r\n'))
            self.img_filename = img_paths
        # read label imgs
        full_labels = []
        if Label_dir:
            label_txt_path = os.listdir(Label_dir)
            for line in label_txt_path:
                with open(Label_dir + line) as ff:
                    pngs = []
                    for lab in ff:
                        pngs.append(lab)
                    full_labels.append(pngs)

        self.label = full_labels
        self.labeli = Label_img_path
        self.img_size = img_size

    def __toMask(self, ldir, idir):
        out = []
        for i, it in enumerate(ldir):
            img = Image.open(idir + it.strip('\r\n'))
            img = img.convert('L')
            out.append(np.array(img.resize((self.img_size, self.img_size))))
        out = np.array(out)
        mask = out.argmax(axis=0)
        return mask

    def __getitem__(self, index):
        img = Image.open(self.img_filename[index])
        img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size))

        if self.label and self.labeli:
            # Read Labels
            mask = np.zeros((self.img_size, self.img_size))
            mask = self.__toMask(self.label[index], self.labeli)

            if self.transform is not None:
                img = self.transform(img)
                mask = torch.from_numpy(mask)
            return img, mask  # shape of img, labels like: [h, w, c], [h, w, labels]
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.img_filename)