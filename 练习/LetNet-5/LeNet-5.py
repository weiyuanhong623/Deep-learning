import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import time

#完整数据集先加载到内存，使用GPU之前从内存拿到显存
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


def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):

    #定制初始化方法
    def init_weight(m):
        #对于全连接层或者卷积层
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

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
    timer,num_batchs=d2l.Timer(),len(train_iter)



    #进行迭代训练
    for epoch in range(num_epochs):
        #训练损失之和、训练准确率之和、样本数
        metric=d2l.Accumulator(3)

        #训练
        net.train()
        for i,(X,y) in enumerate(train_iter):
            timer.start()

            #清空梯度
            optmizer.zero_grad()
            #从内存加载到显存
            X,y=X.to(device),y.to(device)
            #获取输出值
            y_hat=net(X)
            #计算损失（输出和标签）
            l=loss(y_hat,y)
            #反向传播
            l.backward()

            #更新权重
            optmizer.step()

            #计数
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])

            timer.stop()

            train_l=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]

            #画图
            if (i+1)%(num_batchs//5)==0 or i ==num_batchs-1:
                animator.add(epoch+(i+1)/num_batchs,(train_l,train_acc,None))

        test_acc=evaluate_accuracy_gpu(net,test_iter)
        animator.add(epoch+1,(None,None,test_acc))

        print(f'loss {train_l:.3f},train acc {train_acc:.3f},test acc {test_acc:.3f}')
        print(f'{metric[2]*num_epochs/timer.sum():.1f} examples/sec on {str(device)}')


if __name__ == '__main__':
    # 样本数、通道数、高、宽

    # LetNet-5模型
    # 1个输入通道，6对应输出通道数
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2),
                        # 第一个卷积层的输出和输入形状一样（pw、ph=k-1=4）
                        nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        # 第一个池化的输出相比输入高缩小了2倍，宽缩小了2倍，总共缩小了4倍

                        # 6个输入通道，16个输出通道
                        nn.Conv2d(6, 16, kernel_size=5),
                        # 第2个卷积层的输出形状 高：nh=nh-kh+1,宽:nw=nw-kw+1
                        nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        # 第2个池化层的输出相比输入高、宽各缩小2倍

                        # 展平
                        nn.Flatten(),

                        # 全连接块，减少维度
                        nn.Linear(16 * 5 * 5, 120),  # 这里的输入数目要具体分析，这里因为是处理28*28的图像，运算至此展平后为16*5*5
                        nn.Sigmoid(),
                        nn.Linear(120, 84),
                        nn.Sigmoid(),
                        nn.Linear(84, 10)
                        )

    '''
    #随机生成一个样本，一张图片的数据放入模型来查看模型的结构
    #样本的规格是28*28
    X=torch.rand(size=(1,1,28,28),dtype=torch.float32)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,"output shape：",X.shape)

    '''
    # 读取数据集
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    # 训练模型
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())