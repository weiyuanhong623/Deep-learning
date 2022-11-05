import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


#softmax运算
def softmax(X):
    #先对输出求幂
    X_exp=torch.exp(X)

    #求和，得到规范化常数
    #保证原始张量维度（输出的矩阵，每一行为一个样本的输出）
    partition=X_exp.sum(1,keepdim=True)

    #每个求幂后的输出除以规范化常数
    return X_exp/partition

#交叉熵损失函数
#真实标签的预测率的负对数似然
def corss_entropy(y_hat,y):

    #由y记录的真实类别找到y_hat中对于真实类别的预测概率，并拿来求负对数
    return -torch.log(y_hat[range(len(y_hat)),y])


#分类精度
#训练准确率、测试准确率
#accuracy 主要是记录经过softmax运算后的输出的预测结果中正确的数目。下一步是除以总样本数以得到相应的准确率
def accuracy(y_hat,y):

    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)      #对每列求最大值，即求出每个样本的最大预测值下标
    cmp=y_hat.type(y.dtype)==y      #把输出值转化为和标签一样的数据类型

    return float(cmp.type(y.dtype).sum())  # 把比较结果转为y的数据类型并求和


def evaluate_accuracy(net,data_iter):

    #判断net的类型是否是nn.Module（只需要输入模型架构就行）
    if isinstance(net,nn.Module):
        net.eval() #设置为评估模式

    #创建两个变量分别用于存储正确预测数，总预测数
    metric=d2l.Accumulator(2)
    with  torch.no_grad():
        for X,y in data_iter:
            #分别对两个变量累加
            # print(y.numel())
            metric.add(accuracy(net(X),y),y.numel())  #numel()返回每个批量的样本数
            # print(f'测试总数目:{metric[1]}')
    #正确数/总数目
    return metric[0]/metric[1]


def evaluate_accuracy_gpu(net,data_iter,device=None): #@save
    #使用GPU计算模型在数据集上的精度
    if isinstance(net,nn.Module):
        net.eval() #设置为评估模式
        if not device:
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

#一个训练周期
def train_epoch_ch3(net,train_iter,loss,updater):

    #判断net的类型是否是nn.Module（只需要输入模型架构就行）
    if isinstance(net,nn.Module):
        net.train() #设置为训练模式

    #创建3个变量用于存储：总的训练损失，正确预测的数目，样本数
    metric=d2l.Accumulator(3)

    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)

        #判断优化函数是不是Optimizer类型（即使用Pytorch内置的优化器和损失函数）
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            #损失平均值反向传播
            l.mean().backward()
            updater.step()

        else:
            #使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])

        metric.add(float(l.sum()),accuracy(net(X),y),y.numel())

    #训练损失=总训练损失/训练样本数；训练准确率
    return metric[0]/metric[2],metric[1]/metric[2]


def  train_ch3(net,train_iter,test_iter,loss,num_epochs,updator):

    animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend=['train loss','train acc','test acc'])

    #num_epochs变成一个序列
    for epoch in range(num_epochs):

        #获取每个周期
        train_metrics=train_epoch_ch3(net,train_iter,loss,updator)
        test_acc=evaluate_accuracy_gpu(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))

        print(f'epoch{epoch}，train_loss：{train_metrics[0]}；train_acc：{train_metrics[1]}')
    train_loss ,train_acc=train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc





if __name__=='__main__':

    #softmax
    batch_size=256      #迭代器的每次批量大小是256
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)


    num_outpus=10
    num_inputs=784
    #初始化权重和偏差
    #均值为0，方差为0.01的离散正态分布中随机抽取数据
    W=torch.normal(0,0.01,size=(num_inputs,num_outpus),requires_grad=True)
    #偏差设置为0
    B=torch.zeros(num_outpus,requires_grad=True)


    # 定义softmax模型
    def net(X):
        # 先将输入和权重内积，然后加上偏差，接着对输出进行softmax运算。
        #先将输入的展平为向量由于图片格式是28*28，所以向量规格是1*784
        return softmax(torch.matmul(torch.reshape(X,(-1,num_inputs)),W)+B)


    '''
    #测试准确率函数
    test_acc=evaluate_accuracy(net,test_iter)
    print(test_acc)
    '''

    lr =0.1

    def updator(batch_size):
        return d2l.sgd([W, B], lr, batch_size)

    num_epochs=10
    train_ch3(net,train_iter,test_iter,corss_entropy,num_epochs,updator)
    d2l.plt.show()


