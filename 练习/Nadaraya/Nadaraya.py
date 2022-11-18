import torch
from torch import nn
from d2l import torch as d2l



n_train = 50 # 训练样本数
x_train,_= torch.sort(torch.rand(n_train) * 5)      # 返回排序后的训练样本，以及元素对应的下标

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,)) # 训练样本的输出(加上噪音)
#为什么加上,


x_test = torch.arange(0, 5, 0.1) # 测试样本     0-5

y_truth = f(x_test) # 测试样本的真实输出

n_test = len(x_test) # 测试样本数


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
        xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);


# 基于平均汇聚来计算所有训练样本输出值的平均值：
# y_hat=torch.repeat_interleave(y_train.mean(),n_test)
#对全部元素求平均得到一个元素的张量，然后重复50个（恢复到原本张量的大小）


#基于⾮参数的注意⼒汇聚模型

# X_repeat的形状:(n_test,n_train),
# 每⼀⾏都包含着相同的测试输⼊（例如：同样的查询）
# X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))     #先从（50）->50*50=（2500)->(50,50)
# attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)      #相减是广播        x_test是查询      x_train是键
# y_hat = torch.matmul(attention_weights, y_train)        #矩阵乘法   (50,50) （1，50）  得到(1,50)

#批量矩阵乘法实例
# X=torch.ones((2,1,4))
# Y=torch.ones((2,4,6))
# z=torch.bmm(X,Y).shape
# print(z)

#实例+1
# weights = torch.ones((2, 10)) * 0.1
# values = torch.arange(20.0).reshape((2, 10))
#
# print(weights.unsqueeze(1).shape)
# print(values.unsqueeze(-1).shape)
#
# z=torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
# print(z.shape)



#基于带参数的注意力池化
class NWKernelRegression(nn.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        #权重初始化
        self.w=nn.Parameter(torch.rand(1,),requires_grad=True)

    def forward(self,queries,keys,values):

        #将查询数 复制到和keys数目一样，然后变成keys相同的形状
        queries=queries.repeat_interleave(keys.shape[1]).reshape((-1,keys.shape[1]))

        self.attention_weights=nn.functional.softmax(-((queries - keys) * self.w)**2 / 2,dim=1)

        return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)


# X_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输⼊
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))


# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))         #平均每行减少一个

# print(X_tile)
# print(X_tile.shape)
# print((1 - torch.eye(n_train)).type(torch.bool))
# print(keys)
# print(keys.shape)


# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

#均匀分布

# 使⽤平⽅损失函数和随机梯度下降。

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    #清空梯度
    trainer.zero_grad()
    l=loss(net(x_train,keys,values),y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# d2l.plt.show()

# keys的形状:(n_test，n_train)，每⼀⾏包含着相同的训练输⼊（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()


plot_kernel_reg(y_hat)

d2l.plt.show()