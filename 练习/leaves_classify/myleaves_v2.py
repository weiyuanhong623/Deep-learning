import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from d2l import torch as d2l



labels_dataframe = pd.read_csv('../data/leaves/train.csv')
# print(labels_dataframe.index)
# print(labels_dataframe.head(5))
# print(labels_dataframe.describe())

leaves_labels=sorted(list(set(labels_dataframe['label'])))
n_class=len(leaves_labels)
# print(n_class)
# print(leaves_labels[:10])
#
class_to_num=dict(zip(leaves_labels,range(n_class)))
# print(class_to_num)



class leavesData(Dataset):
    def __init__(self, csv_path, file_path, mode, valid_ratio=0.2):

        self.data_info = pd.read_csv(csv_path, header=None)  #header=None是去掉表头部分
        self.data_len=len(self.data_info.index)-1         #.index获取一个 range类型的索引
        self.train_len = int(self.data_len * (1 - valid_ratio))
        self.file_path=file_path
        self.mode=mode

        if mode == 'train':
            # 第一列包含图像文件的名称       #asarray不会占用新内存
            #获取train集的图像名称
            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label

        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label

        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        #获取读取图片的长度
        self.real_len = len(self.image_arr)
        print(f'Finished reading the {mode} set of Leaves Dataset ({self.real_len} samples found)')

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len

#读取图片数据
train_path = '../data/leaves/train.csv'
test_path = '../data/leaves/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = '../data/leaves/'

train_dataset = leavesData(train_path, img_path, mode='train')
val_dataset = leavesData(train_path, img_path, mode='valid')
test_dataset = leavesData(test_path, img_path, mode='test')
# print(train_dataset)
# print(val_dataset)
# print(test_dataset)


# 定义data loader     批量大小是64
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=3
    )

val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=3
    )
test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=3
    )



#残差块
class Residual(nn.Module):
    #输入通道数，输出通道数
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()

        #卷积层

        #第一层卷积层
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)

        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)

        if use_1x1conv:
            #即对于x使用1x1卷积层
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None

        #归一化层
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

    #定义正向传播                                 #整体是在ReLU激活函数之前进行残差映射或者恒等映射
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))       #先经过第一层卷积层然后进行归一化，接着进行ReLU激活函数
        Y=self.bn2(self.conv2(Y))               #接着进入第二层卷积层然后归一化
        if self.conv3:                          #对于x进行1x1卷积层或者不进行任何操作。
            X=self.conv3(X)
        Y+=X
        return  F.relu(Y)                       #映射后进行激活


def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk=[]

    #残差块数目
    for i in range(num_residuals):

        #对第一个残差块的x使用1x1卷积层
        if i ==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))

        #其他残差块的x不做变换
        else:
            blk.append(Residual(num_channels,num_channels))

    return blk


def get_models():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 第一个模块中的第一个残差块的x不使用1x1卷积层
    b3 = nn.Sequential(*resnet_block(64, 128, 2))  # 使用1x1卷积层
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, n_class))
    return net




def evaluate_accuracy_gpu(net,data_iter,device=None): #@save
    #使用GPU计算模型在数据集上的精度
    if isinstance(net,nn.Module):
        net.eval() #设置为评估模式
        if not device:
            #若未指明训练设备则获取net所在的设备进行训练
            device=next(iter(net.parameters())).device

    metric=d2l.Accumulator(2)

    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                #BERT微调所需的
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)

            y=y.to(device)
            metric.add(d2l.accuracy(net(X),y),y.numel())

    return metric[0]/metric[1]


def train_leaves(model,num_epochs,lr,device):

    #定制初始化方法
    def init_weight(m):
        #对于全连接层或者卷积层
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)       #根据输入、输出的大小进行初始化使得输入和输出的方差大小差不多。

    model.apply(init_weight)

    print('training on ',device)
    #将网络结构转移到显存中
    net.to(device)

    #使用SGD优化器
    optmizer=torch.optim.SGD(net.parameters(),lr=lr)

    #使用交叉熵损失函数
    loss=nn.CrossEntropyLoss()

    #记录训练用时和样本总数
    timer1,timer2,num_batchs=d2l.Timer(),d2l.Timer(),len(train_loader)
    #总训练用时
    timer1.start()


    for epoch in range(num_epochs):
        model.train()

        metric=d2l.Accumulator(3)       #创建两个存储训练误差和训练准确率的变量    以及样本数
        for i,(imgs,labels) in enumerate(train_loader):
            timer2.start()

            #清空梯度
            optmizer.zero_grad()
            #从内存加载到显存
            imgs,labels=imgs.to(device),labels.to(device)

            #获取输出值
            y_out=model(imgs)

            #计算损失（输出和标签）
            l=loss(y_out,labels)             #在这里已经完成softmax的运算

            #反向传播
            l.backward()

            #更新权重
            optmizer.step()

            with torch.no_grad():
                metric.add(l*imgs.shape[0],d2l.accuracy(y_out,labels),imgs.shape[0])   #使用内置的损失函数求得的是损失均值，对于每个批量需要乘以数目求出总和，然后对每个批量的损失进行累加。
            timer2.stop()



            train_l=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]

            #验证准确率
            valid_acc = evaluate_accuracy_gpu(net,val_loader)

        print(f'loss {train_l:.3f},train acc {train_acc:.3f},test acc {valid_acc:.3f}')            #
        print(f'{metric[2]*num_epochs/timer2.sum():.1f} examples/sec on {str(device)}')

    timer1.stop()
    print("total time: ",timer1.times[0]/60," min")



if __name__=='__main__':

    net=get_models()

    num_epochs,lr=10,0.05

    train_leaves(net,num_epochs,lr,d2l.try_gpu())





