import os
import sys
import random

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
from torch.utils import data
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchvision.datasets as datasets

data_path="./data/I2G"


class DataLoaderHelper(data.Dataset):
    def __init__(self, root,input_image_size):
        super(DataLoaderHelper, self).__init__()
        self.data_transform=transforms.Compose([
            transforms.RandomResizedCrop(input_image_size)
        ])
        self.data_tensor_transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform=transforms.Compose([
            transforms.Resize((16,16)),
        ])
        self.mask_tensor_transform=transforms.Compose([transforms.ToTensor()])
        root_mask=root+"_mask"
        self.mask_imgs=[]
        self.imgs=[]
        mask_imgs=os.listdir(root_mask)
        imgs=os.listdir(root)
        for k in mask_imgs:
            dir=os.path.join(root_mask,k)
            for l in os.listdir(dir):
                self.mask_imgs.append(os.path.join(dir,l))
        for k in imgs:
            dir=os.path.join(root,k)
            for l in os.listdir(dir):
                self.imgs.append(os.path.join(dir,l))


    def __getitem__(self, index):
        img_path=self.imgs[index]
        mask_path=self.mask_imgs[index]
        #real-> 1 fake ->0
        label=1 if 'real' in img_path.split('/')[-1] else 0
        pil_img=Image.open(img_path)
        mask_img=Image.open(mask_path)
        if self.data_transform:
            data=self.data_transform(pil_img)
            mask=self.mask_transform(mask_img)
            if random.random()<=0.5:
                data=TF.hflip(data)
                mask=TF.hflip(mask)
            data=self.data_tensor_transform(data)
            mask=self.mask_tensor_transform(mask)
            #mask=self.data_transform(mask_img)
        else:
            pil_img=np.asarray(pil_img)
            data=torch.from_numpy(pil_img)
        return data,label,mask

    def __len__(self):
        return len(self.imgs)

class DataLoaderVHelper(data.Dataset):
    def __init__(self, root,input_image_size):
        super(DataLoaderVHelper, self).__init__()
        self.data_transform=transforms.Compose([
            transforms.RandomResizedCrop(input_image_size)
        ])
        self.data_tensor_transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.imgs=[]
        self.video_names=[]
        imgs=os.listdir(root)
        for k in imgs:
            dir=os.path.join(root,k)
            for l in os.listdir(dir):
                self.imgs.append(os.path.join(dir,l))
                self.video_names.append(l)


    def __getitem__(self, index):
        img_path=self.imgs[index]
        vi_name=self.video_names[index]
        vid=vi_name.split('_')
        video_name=int(vid[0])
        #print(vid[0])
        #real-> 1 fake ->0
        label=1 if 'real' in img_path.split('/')[-1] else 0
        pil_img=Image.open(img_path)
        if self.data_transform:
            data=self.data_transform(pil_img)
            if random.random()<=0.5:
                data=TF.hflip(data)
            data=self.data_tensor_transform(data)
        else:
            pil_img=np.asarray(pil_img)
            data=torch.from_numpy(pil_img)
        return data,label,video_name

    def __len__(self):
        return len(self.imgs)

class Config(object):
    log = './log'  # Path to save log
    checkpoint_path = './checkpoints'  # Path to store checkpoint model
    resume = './checkpoints/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    test= './checkpoints/latest_pcl_demo.pth'
    video_test = None#'./checkpoints/latest_pcl+I2G.pth'
    train_dataset_path = os.path.join(data_path, 'train')
    val_dataset_path = os.path.join(data_path, 'eval')
    test_dataset_path = './test_data/DFD'

    network = "resnet34"
    pretrained = True
    num_classes = 2
    seed = 0
    input_image_size = 256
    scale = 256 / 256
    loss_lambda = 10

    train_dataset=DataLoaderHelper(train_dataset_path,input_image_size)
    # train_dataset = datasets.ImageFolder(
    #     train_dataset_path,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(input_image_size),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225]),
    #     ]))
    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    
    test_dataset = datasets.ImageFolder(
        test_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    video_test_dataset=DataLoaderVHelper(test_dataset_path,input_image_size)

    milestones = [30, 60, 90]
    epochs = 100
    batch_size = 128
    accumulation_steps = 1
    lr = 5*(1e-5)
    weight_decay = 1e-4
    momentum = 0.9
    num_workers = 0
    print_interval = 100
    apex = False
