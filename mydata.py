#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *


class Mydata(Dataset):
    def __init__(self, rootpth, cropsize=(640, 640), mode='train',
                 randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        # super(CityScapes, self).__init__(*args, **kwargs)
        super(Mydata, self).__init__()
        assert mode in ('train', 'val', 'test',"evaluate")
        # train val 分别指训练时候的训练集和验证集；test指只输入图片预测；evaluate是指做指标评估时，同时输入图片和标签图，但是不做增强
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        with open('./cityscapes_info-mydata.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info if
                       (el['ignoreInEval'] != 'false' and el['trainId'] != 255)}  # 创建  映射字典 {序号：像素值}
        # mode的区别 test就可以只输入图片不输入标签，trainval就是两个都有
        if self.mode == 'train' or self.mode == 'val'or self.mode =="evaluate":
            ## parse img directory
            self.imgs = {}
            imgnames = []
            impth = osp.join(rootpth, 'imgs', mode)
            folders = os.listdir(impth)
            for fd in folders:
                fdpth = osp.join(impth, fd)
                im_names = os.listdir(fdpth)
                names = [el.replace('.bmp', '') for el in im_names]
                impths = [osp.join(fdpth, el) for el in im_names]
                imgnames.extend(names)
                self.imgs.update(dict(zip(names, impths)))

            ## parse gt directory
            self.labels = {}
            gtnames = []
            gtpth = osp.join(rootpth, 'labels', mode)
            folders = os.listdir(gtpth)
            for fd in folders:
                fdpth = osp.join(gtpth, fd)
                lbnames = os.listdir(fdpth)
                names = [el.replace('.bmp', '') for el in lbnames]
                lbpths = [osp.join(fdpth, el) for el in lbnames]
                gtnames.extend(names)
                self.labels.update(dict(zip(names, lbpths)))

            self.imnames = imgnames
            self.len = len(self.imnames)
            print('self.len', self.mode, self.len)
            assert set(imgnames) == set(gtnames)  # 断言 两个集合是否相等，如果内容相同但是顺序不同也会断言成功。断言成功才执行下一语句
            assert set(self.imnames) == set(self.imgs.keys())
            assert set(self.imnames) == set(self.labels.keys())
        else:
            ## parse img directory
            self.imgs = {}
            imgnames = []
            impth = osp.join(rootpth, 'imgs', mode)
            folders = os.listdir(impth)
            for fd in folders:
                fdpth = osp.join(impth, fd)
                im_names = os.listdir(fdpth)
                names = [el.replace('.bmp', '') for el in im_names]
                impths = [osp.join(fdpth, el) for el in im_names]
                imgnames.extend(names)
                self.imgs.update(dict(zip(names, impths)))
            self.imnames = imgnames
            self.len = len(self.imnames)
            print('self.len', self.mode, self.len)
            assert set(self.imnames) == set(self.imgs.keys())


        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
        ])

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val'or self.mode =="evaluate":
            fn = self.imnames[idx]
            impth = self.imgs[fn]
            lbpth = self.labels[fn]
            img = Image.open(impth).convert('RGB')
            label = Image.open(lbpth)
            if self.mode == 'train' or self.mode == 'val':
                im_lb = dict(im=img, lb=label)
                #做推理的时候不需要数据增强
                im_lb = self.trans_train(im_lb)
                img, label = im_lb['im'], im_lb['lb']
            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
            label = self.convert_labels(label)
            # label = np.squeeze(label)
            # 此处发现，在调用torch._C._nn.cross_entropy_loss()函数的时候，label应该是（640,640），如果没有这个语句，label是（1,640,640），在此网络中没有出现，再别的网络出现
            return img, label

        else:
            fn = self.imnames[idx]
            impth = self.imgs[fn]
            img = Image.open(impth).convert('RGB')
            img = self.to_tensor(img)
            return img

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


if __name__ == "__main__":
    from tqdm import tqdm

    ds = Mydata('E:\project\cvmart_STDC-main0\cityscapes/', n_classes=19, mode='test')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))
