import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


#残差块
class Residual(nn.Module):
    #输入通道数，输出通道数
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()

        #卷积层
        #（输入输出形状不变）

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


if __name__=='__main__':

    # reblk=Residual(3,3) #输入通道数3，输出3
    # X=torch.rand(4,3,6,6)
    # print(X)
    # Y=reblk(X)          #进行正向传播
    # print(Y.shape)
    # print(Y)
    # #或者增加输出通道数的同时减半输出高、宽
    # # blk = Residual(3, 6, use_1x1conv=True, strides=2)

