import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F



class MLP(nn.Module):
    def __init__(self):
        #调用MLP的父类Module的构造函数来进行初始化
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)

    def forward(self,X):
        return self.out(nn.functional.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx,module in enumerate(args):
            #module是Module子类的一个实例。保存在Module类的成员变量的_modules中。类型是OrderedDict
            self._modules[str(idx)]=module

    def forward(self,X):
        ## OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X=block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()

        #向前传播不需要计算梯度
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        self.linear=nn.Linear(20,20)

    #前向传播
    def forward(self,X):
        #全连接层
        X=self.linear(X)

        #隐藏层 relu使用创建的常量参数（这里是随机权重+1）
        X=F.relu(torch.mm(X,self.rand_weight)+1)

        #全连接层
        X=self.linear(X)

        while X.abs().sum()>1:
        #在L1范数⼤于1的条件下，将输出向量除以2，直到它满⾜条件为⽌。
            X/=2
        return X.sum()




class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear=nn.Linear(32,16)

    #定义向前出传播
    def forward(self,X):
        #先按顺序将数据前向传播进入net的每一层，接着传播到全连接层输出。
        return self.linear(self.net(X))




X=torch.rand(2,20);
print(X)

#
#net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))


#net(X)
#

net=NestMLP()

print(net(X))