import math
import collections
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F



# d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
# '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
# # 如果你使⽤完整的Kaggle竞赛的数据集，设置demo为False
# demo = True
# if demo:
#     data_dir = d2l.download_extract('cifar10_tiny')
# else:
#     data_dir = '../data/cifar-10/'





def read_csv_labels(fname):

    with open(fname, 'r') as f:
        # 跳过⽂件头⾏(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]     #删除末尾空格，按照','分隔,转为列表
    print(tokens)
    print(type(tokens))
    return dict(((name, label) for name, label in tokens))

data_dir='../data/kaggle_cifar10_tiny/'

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv')) #将.csv文件转化为字典，序号是key 标签是值
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))      #标签去重后获取长度


def copyfile(filename, target_dir):
#将⽂件复制到⽬标⽬录
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


#一共创建了3个文件夹：train_valid_test中的train_valid、train、valid
def reorg_train_valid(data_dir, labels, valid_ratio):
    #"""将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数

    # 首先按照字典的值进计数，得到标签为key，数目为值的字典 然后按照数目进行降序排列后    获取样本数最少的键对值的值(0是key，1是值)
    n = collections.Counter(labels.values()).most_common()[-1][1]

    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):      #train中就是训练集图片
        label = labels[train_file.split('.')[0]]                        #首先获取训练图的序号（从1开始）   然后获取对应图片的标签
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test','train_valid', label))        #把图片复制到对应类别的文件夹中
        if label not in label_count or label_count[label] < n_valid_per_label:          #未参与计数或者该类别的图片数目未达标
            copyfile(fname, os.path.join(data_dir, 'train_valid_test','valid', label))  #就添加到验证集
            label_count[label] = label_count.get(label, 0) + 1                          #对标签数目进行计数（get的意思是若不存在就设值为0）
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test','train', label))  #若其他部分添加到训练集
    #返回验证机数
    return n_valid_per_label


#在train_valid_test中创建test
def reorg_test(data_dir):
#在预测期间整理测试集，以⽅便读取
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),os.path.join(data_dir, 'train_valid_test', 'test','unknown'))


def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


# 批量大小和分割系数
batch_size = 32
valid_ratio = 0.1
#数据预处理：整理好test、train、valid、train_valid数据集
reorg_cifar10_data(data_dir, valid_ratio)


#训练集的图像增广
transform_train = torchvision.transforms.Compose([
    # 在⾼度和宽度上将图像放⼤到40像素的正⽅形
    torchvision.transforms.Resize(40),

    # 随机裁剪出⼀个⾼度和宽度均为40像素的正⽅形图像，
    # ⽣成⼀个⾯积为原始图像⾯积0.64到1倍的⼩正⽅形，
    # 然后将其缩放为⾼度和宽度均为32像素的正⽅形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),

    #rgb转为张量
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

#测试集的图像增广
# 在测试期间，我们只对图像执⾏标准化，以消除评估结果中的随机性。
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
    [0.2023, 0.1994, 0.2010])])

#数据集读取


#分别返回训练、总的数据集和对应标签
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test/', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

#分别返回验证、测试的数据集和对应标签
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test/', folder),
    transform=transform_test) for folder in ['valid', 'test']]

#迭代器
#转换为列表（因为是两个一起转化的原因？）
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,drop_last=True)

#测试集不引入图像变换
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,drop_last=False)


#定义模型


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


#类别数，输入通道
num_class=10
channel_input=3

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
    b1 = nn.Sequential(nn.Conv2d(channel_input, 64, kernel_size=7, stride=2, padding=3),
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
                        nn.Linear(512, num_class))
    return net
loss = nn.CrossEntropyLoss(reduction="none")


#每个批量的训练
def train_batch(net, X, y, loss, trainer, devices):

    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


#训练
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay):

    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    num_batches, timer,timer2 = len(train_iter), d2l.Timer(),d2l.Timer()
    legend = ['train loss', 'train acc']

    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    timer2.start()

    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()

            l, acc = train_batch(net, features, labels,loss, trainer, devices)

            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))

        if valid_iter is not None:
            valid_acc =evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()

    timer2.stop()
    print("total time: ",timer2.times[0]/60," min")

    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                    f'train acc {metric[1] / metric[2]:.3f}')

    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                         f' examples/sec on {str(devices)}')








#优化算法的学习速率将在每4个周期乘以0.9
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_models()

train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay)

d2l.plt.show()




