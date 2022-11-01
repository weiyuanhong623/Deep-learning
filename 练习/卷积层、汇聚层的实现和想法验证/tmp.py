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


#卷积运算
def corr2d(X,K): #@save
    #获取卷积核高、宽
    h,w=K.shape
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):             #根据输出的规格来按照元素进行卷积运算。就能够实现对于输入的部分元素的舍弃（不进行卷积运算）
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

#卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super.__init__()
        self.weight=nn.Parameter(torch.rand(kernel_size))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,X):
        return corr2d(X,self.weight)+self.bias


def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    print(X.shape)
    Y=conv2d(X)
    print(Y.shape)

    #忽略前两个维度
    return Y.reshape(Y.shape[2:])
    # return Y

#多通道输入，单通道输出
def corr2d_multi_in(X,K):
    #按照通道维度进行卷积运算（互相关运算），再相加
    return sum(corr2d(X,K) for X,K in zip(X,K))

#多通道输出
def corr2d_multi_in_out(X,K):
    #按照K的通道维度对输入X进行互相关运算，最后进行叠加
    # for k in K:
    #     print(k)

    return torch.stack([corr2d_multi_in(X,k) for k in K],0)


def corr2d_multi_in_out_1x1(X,K):
    #输入通道，高、宽
    c_i,h,w=X.shape
    #输出通道
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))

    #矩阵乘法
    Y=torch.matmul(K,X)
    return Y.reshape((c_o,h,w))

def pool2d(X,pool_size,mode='max'):
    p_h,p_w=pool_size
    Y=torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='max':
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode=='avg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
    return Y

if __name__=='__main__':


    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    X=torch.cat((X,X+1),1)
    print(X.shape)
    print(X)
    mypool2d=nn.MaxPool2d(3,padding=1,stride=2)
    result=mypool2d(X)
    print(result)
    # #池化
    # X=torch.arange(9.)
    # X=X.reshape((3,3))
    # result=pool2d(X,(2,2))
    # print(result)
    #
    # result2=pool2d(X,(2,2),'avg')
    # print(result2)
    #
    # X2=torch.ones((6,8))
    # X2[:,2:6]=0
    # K=torch.tensor([[1.0,-1.0]])
    # result3=corr2d(X2,K)
    # print(result3)
    # result4=pool2d(result3,(2,2))
    # print(result4)

    # X=torch.normal(0,1,(3,3,3))
    # #输出通道数是2,输入通道数是3
    # K=torch.normal(0,1,(2,3,1,1))
    # Y1=corr2d_multi_in_out(X,K)
    # Y2=corr2d_multi_in_out_1x1(X,K)
    # print(Y1)
    # print(Y2)
    # assert float(torch.abs(Y1-Y2).sum())< 1e-6

    # X=torch.arange(18)
    # #通道数2
    # X=X.reshape((2,3,3))
    # K=torch.arange(8)
    # K=K.reshape((2,2,2))
    # K=K+1
    # result=corr2d_multi_in(X,K)
    # print(result)
    # # print(K)
    # #输出通道数为3
    # K=torch.stack((K,K+1,K+2),0)
    # # print(K.shape)
    # # print(K)
    #
    # result=corr2d_multi_in_out(X,K)
    # print(result)

    #
    # result=corr2d_multi_in(X,K)
    # print(result)

    # X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
    #                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    #
    # print(X.shape)
    # K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    # print(K.shape)

    # myconv2d=nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
    # X=torch.rand(size=(8,8))
    # comp_conv2d(myconv2d,X).shape


    # X=torch.arange(9)
    # # print(X)
    # X=X.reshape((3,3))
    # K=torch.arange(4)
    # K=K.reshape((2,2))
    # Y=corr2d(X,K)
    # print(Y)
    # #批量大小，通道数、高度、宽度
    # myconv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
    #
    # X=X.reshape((1,1,6,8))
    # Y=Y.reshape((1,1,6,7))
    # #学习率
    # lr=3e-2
    #
    # for i in range(10):
    #     Y_hat=myconv2d(X)
    #     #平方损失
    #     l=(Y_hat-Y)**2
    #
    #     #梯度清零
    #     myconv2d.zero_grad()
    #     l.sum().backward()
    #
    #     myconv2d.weight.data[:]-=lr*myconv2d.weight.grad
    #     if (i+1)%2==0:
    #         print(f'epoch {i+1},loss {l.sum():.3f}')
    #
    #
    #     print(myconv2d.weight.data)


    # #验证数据流动：
    # X=torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]])
    # # Y=torch.randn(size=(1,4))
    # # print(Y)
    # print(X.shape)
    #
    # net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,2))
    # #权重初始化为1,偏置初始化为0
    # net.apply(init_constant)
    # # net(X)
    # print(net)
    # print(net(X))
    # print(net[0].weight.data, net[0].bias.data)
    # print(net[2].weight.data, net[2].bias.data)
    #
    # print(len(net))


    # X = torch.rand((10000, 10000))
    # Y = X.cuda(0)
    # time_start = time.time()
    # Z = torch.mm(X, X)
    # time_end = time.time()
    # print(f'cpu time cost: {round((time_end - time_start) * 1000, 2)}ms')
    # time_start = time.time()
    # Z = torch.mm(Y, Y)
    # time_end = time.time()
    # print(f'gpu time cost: {round((time_end - time_start) * 1000, 2)}ms')


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

