import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import time



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



# X=torch.rand(size=(2,4))

#
#net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))


#




#参数访问


#嵌套模块
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())


def block2():
    net=nn.Sequential()
    for i in range(4):

        #每个模块的名字为block+i,block2模块一共添加了4个模块；每个模块具体内容为block1，
        net.add_module(f'block{i}',block1())
    return net


# rgnet=nn.Sequential(block2(),nn.Linear(4,1))




def init_normal(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.1)
        nn.init.zeros_(m.bias)



def init_constant(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)



def init_xavier(m):
    if type(m)==nn.Linear:
        nn.init.xavier_normal_(m.weight)

def inin_42(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,42)

# net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
# net[0].apply(init_xavier)
# print(net[0].weight.shape,net[0].bias.shape)
# net[2].apply(inin_42)
# print(net[2].weight.shape,net[2].weight.data,net[2].bias.shape)


def my_init(m):
    if type(m)==nn.Linear:
        print("Init",*[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        print(m.weight)
        m.weight.data*=m.weight.data.abs()>=5

# net.apply(my_init)
# print(net[0].weight[:2])


#参数绑定（共享参数）
# shared=nn.Linear(8,8)
# net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
# print(net)
# net(X)

#不带参数的层

class CenteredLayer(nn.Module):
    def __init__(self):
        #调用父类的构造函数
        super().__init__()

    def forward(self,X):
        return X-X.mean()

# layer=CenteredLayer()
# result=layer(torch.FloatTensor([1,2,3,4,5]))
# print(result)
#
# 加入到其他层中
# net=nn.Sequential(nn.Linear(4,8),CenteredLayer())
# print(net(X))
# print(net(X).mean())


class MyLinear(nn.Module):
    #输入数、输出数
    def __init__(self,in_units,units):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(in_units,units))
        self.bias=nn.Parameter(torch.randn(units,))

    #定义向前传播
    def forward(self,X):
        linear=torch.matmul(X,self.weight.data)+self.bias
        return F.relu(linear)


# linear=MyLinear(4,2)
# print(linear.weight)


# print(linear(X))


# net=MySequential(MyLinear(4,8),nn.Linear(8,2))
# print(net(X))

#张量的保存
X = torch.arange(4)
y = torch.zeros(4)
# print([x,y])
# torch.save(x,'x-file')

# 不存在gpu的时候运行代码
def try_gpu(i=0):  #@save

    # 如果存在gpu则返回gpu(i)，否则返回cpu
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus(): #@save

    #返回所有可用的gpu，如果没有gpu则返回cpu()
    devices=[torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else  [torch.device('cpu')]


if __name__=='__main__':
    X = torch.rand((10000, 10000))
    Y = X.cuda(0)
    time_start = time.time()
    Z = torch.mm(X, X)
    time_end = time.time()
    print(f'cpu time cost: {round((time_end - time_start) * 1000, 2)}ms')
    time_start = time.time()
    Z = torch.mm(Y, Y)
    time_end = time.time()
    print(f'gpu time cost: {round((time_end - time_start) * 1000, 2)}ms')

    # a=torch.tensor([1,2,3],device=try_gpu())
    # print(a)
    # print(a.device)
    # torch.save([x,y],'x-file')
    # x2,y2=torch.load('x-file')
    # print(x2,y2)

    # mydict={'x':x,'y':y}
    # torch.save(mydict,'mydict')
    # mydict2=torch.load('mydict')
    # print(mydict2)


    # clone=MLP()
    # clone.load_state_dict(torch.load('mlp_params'))
    # Y_clone=clone(X)
    # print(torch.cuda.device_count()

    # print(torch.device('cuda'))
    # print(try_gpu())
    # print(try_gpu(1))
    # print(try_all_gpus())