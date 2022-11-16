import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import KFold
from torch.nn import functional as F

from d2l import torch as d2l





#加载数据
train_csv=pd.read_csv('../data/leaves/train.csv')

#获取训练集标签
leaves_labels=sorted(list(set(train_csv['label'])))     #获取标签并按字母排序

#类别数
n_classes=len(leaves_labels)

#将序号和类别打包成一个元组
class_to_num=dict(zip(leaves_labels,range(n_classes)))  #按照字母排序将标签转换为one_h


#继承Dataset
class ReadData(data.Dataset):
    def __init__(self,cvs_data,transform=None):
        super(ReadData,self).__init__()
        self.data=cvs_data
        self.transform=transform

    #重写魔法方法 index
    def __getitem__(self, idx):

        #将路径和对应图片的名称平起来 获得完整路径
        img=Image.open("../data/leaves/"+self.data.loc[idx,"image"])

        #获取训练集的样本标签
        label=class_to_num[self.data.loc[idx,"label"]]      #内层获取的是标签名称,通过外层转换为one_hot
        return img,label

    #重写魔法方法
    def __len__(self):
        return len(self.data)

def load_leaves(dataset,k,batch_size,resize=None):

    #图像转换(这里是”兼容卡“)
    trans=[transforms.ToTensor(),  #把图像转换为张量
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
           ]
    if resize:
        trans.insert(0,transforms.Resize(resize))   #改变图像大小(插入列表)
    trans=transforms.Compose(trans)

    KF=KFold(n_splits=k)
    for train_idxs,test_idxs in KF.split(dataset):          #共获得k组下标，每次一组(每次包含train_test)
        train_data=data.loc[train_idxs].reset_index(drop=True)
        valid_data=data.loc[test_idxs].reset_index(drop=True)

        train_iter=data.DataLoader(
            ReadData(valid_data, trans),
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            # pin_memory=True   锁业内存
        )

        valid_iter=data.DataLoader(
            ReadData(valid_data,trans),
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            # pin_memory=True
        )

        yield train_iter,valid_iter



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


def get_models(k=5):
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
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
                        nn.Linear(512, n_classes))
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



def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):

    #定制初始化方法
    def init_weight(m):
        #对于全连接层或者卷积层
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)       #根据输入、输出的大小进行初始化使得输入和输出的方差大小差不多。

    #进行初始化
    net.apply(init_weight)

    print('training on ',device)
    #将网络结构转移到显存中
    net.to(device)

    #使用SGD优化器
    optmizer=torch.optim.SGD(net.parameters(),lr=lr)

    #使用交叉熵损失函数
    loss=nn.CrossEntropyLoss()

    #可视化（x坐标是迭代周期，设置比例、设置图示）
    animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
                          legend=['train loss','train acc','test acc'])

    #记录训练用时和样本总数
    timer1,timer2,num_batchs=d2l.Timer(),d2l.Timer(),len(train_iter)

    #总训练用时
    timer1.start()

    #进行迭代训练
    for epoch in range(num_epochs):

        # for name,param in net.named_parameters():
        #     print(name)
        #     print(param.data)
        #     print("requires_grad:", param.requires_grad)
        #     print("-----------------------------------")

        #训练损失之和、训练准确率之和、样本数
        metric=d2l.Accumulator(3)

        #训练
        net.train()
        for i,(X,y) in enumerate(train_iter):
            timer2.start()

            #清空梯度
            optmizer.zero_grad()
            #从内存加载到显存
            X,y=X.to(device),y.to(device)

            #获取输出值
            y_hat=net(X)
            #计算损失（输出和标签）
            l=loss(y_hat,y)             #在这里已经完成softmax的运算
            #反向传播
            l.backward()

            #更新权重
            optmizer.step()

            #计数
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])   #使用内置的损失函数求得的是损失均值，对于每个批量需要乘以数目求出总和，然后对每个批量的损失进行累加。

            timer2.stop()

            train_l=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]

            #画图
            if (i+1)%(num_batchs//5)==0 or i ==num_batchs-1:
                animator.add(epoch+(i+1)/num_batchs,(train_l,train_acc,None))

        test_acc=evaluate_accuracy_gpu(net,test_iter)
        animator.add(epoch+1,(None,None,test_acc))

        print(f'loss {train_l:.3f},train acc {train_acc:.3f},test acc {test_acc:.3f}')
        print(f'{metric[2]*num_epochs/timer2.sum():.1f} examples/sec on {str(device)}')
    timer1.stop()

    print("total time: ",timer1.times[0]/60," min")