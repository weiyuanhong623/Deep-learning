import torch
from torch import nn,optim
from  torch.utils.data import *
from torchvision import transforms
from d2l import torch as d2l
import timm

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time




# train_data=pd.read_csv('../data/leaves/train.csv')
# test_data=pd.read_csv('../data/leaves/test.csv')
# print(test_data.shape)

#加载数据
train_csv=pd.read_csv('../data/leaves/train.csv')

# x1=set(train_data['label'])
# x=list(set(train_data['label']))
# print(x1)
# print(x)

#获取训练集标签
leaves_labels=list(set(train_csv['label']))

#类别数
n_classes=len(leaves_labels)

#将序号和类别打包成一个元组
class_to_num=dict(zip(leaves_labels,range(n_classes)))

#获取图片名称
# x=train_csv.loc[4,'image']
# print(x)
# print(train_csv.iloc[0:4,:])


#继承Dataset
class ReadData(Dataset):
    def __init__(self,cvs_data,transform=None):
        super(ReadData,self).__init__()
        self.data=cvs_data
        self.transform=transform

    #重写魔法方法 index
    def __getitem__(self, idx):

        #将路径和对应图片的名称平起来 获得完整路径
        img=Image.open("../data/leaves/"+self.data.loc[idx,"image"])

        #获取训练集的样本标签
        label=class_to_num[self.data.loc[idx,"label"]]
        return img,label

    #重写魔法方法
    def __len__(self):
        return len(self.data)

#图像变换
#训练集
train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=.5),      #依概率p水平翻转
    transforms.ToTensor(),                      #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])  #标准化一张张量图片，有RGB三个通道，对每个通道都进行标准化
])

#验证集
valid_tranform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])  #标准化
])


#k折交叉验证的数据分割
#获取每次验证的数据集
def kfold(data,k=5):

    KF=KFold(n_splits=k,shuffle=False)

    for train_idxs,test_idxs in KF.split(data):
        train_data=data.loc[train_idxs].reset_index(drop=True)
        valid_data=data.loc[test_idxs].reset_index(drop=True)

        train_iter=DataLoader(
            ReadData(train_data,train_transform),
            batch_size=64,
            shuffle=True,
            num_workers=3,
            pin_memory=True     #锁页内存       更快的从内存转移到显存
        )


        valid_iter=DataLoader(
            ReadData(valid_data,valid_tranform),
            batch_size=64,
            shuffle=True,
            num_workers=3,
            pin_memory=True
        )

    yield train_iter,valid_iter


#数据增强

#随机叠加两张图片
def mixup_data(x,y,alpha=1.0,use_cuda=True):
    if alpha>0:
        lam=np.random.beta(alpha,alpha)
    else:
        lam=1

    batch_size=x.size()[0]
    if use_cuda:
        index=torch.randprem(batch_size).cuda()
    else:
        index=torch.randperm(batch_size)

    mixed_x=lam*x+(1-lam)*x[index,:]
    y_a,y_b=y,y[index]

    return mixed_x,y_a,y_b,lam

#图像颜色，亮度，色调等
def color(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    new = transforms.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    new_x = new(x)
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam

#对图像进行反转
def flip_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    new = transforms.RandomRotation(degrees=(90, 180))      #90-180°之内随机翻转
    new_x = new(x)
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


#随机裁剪
def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """ Cutmix 数据增强 -> 随机对主图像进行裁剪, 加上噪点图像
    W: 添加裁剪图像宽
    H: 添加裁剪图像高
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


def get_models(k=5):
    models={}
    for mk in range(k):
        model=timm.create_model("resnest50d_4s2x40d",False,drop_rate=.5)

        #全连接
        model.fc=nn.Sequential(nn.Linear(model.fc.in_features,512),
                               nn.ReLU(inplace=True),
                               nn.Dropout(.3),
                               nn.Linear(512,len(class_to_num))
                               )

        model.load_state_dict(torch.load("resnest50dnew/Resnest50d_new.pth"))
        for i, param in enumerate(model.children()):
            if i == 6:
                break
            param.requires_grad = False

        model.cuda()

        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 10, T_mult=2)
        models[f"model_{mk}"] = {
            "model": model,
            "opt": opt,
            "scheduler": scheduler,
            "last_acc": .97
        }

    return models

if __name__=='__main__':
    models=get_models()