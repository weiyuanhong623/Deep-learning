import torch
from torch import nn
from d2l import  torch as d2l



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



#卷积块
def conv_block(input_channels,num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels),
                         nn.ReLU(),
                         #输入输出不变
                         nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1))


class DenseBlock(nn.Module):
    def __init__(self,num_convs,input_channels,num_channels):
        super(DenseBlock,self).__init__()

        layer=[]
        for i in range(num_convs):

            #每个Dense块由多个卷积块组成，每个卷积块的输出通道数相同
            layer.append(conv_block(num_channels*i+input_channels, num_channels))
        self.net=nn.Sequential(*layer)

    #定义前向传播
    #将每个卷积块的输入和输出在通道维度上连结（可在前面卷积块的通道设置上看出）
    def forward(self,X):
        for blk in self.net:
            Y=blk(X)
            X=torch.cat((X,Y),dim=1)

        return X



def transition_block(input_channels,num_channels):
    return nn.Sequential(nn.BatchNorm2d(input_channels),
                         nn.ReLU(),
                         nn.Conv2d(input_channels,num_channels,kernel_size=1),
                         #输出高、宽减半
                         nn.AvgPool2d(kernel_size=2,stride=2))




#ResNet模型

b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),      #（-7+2+6）/2
                 nn.BatchNorm2d(64),
                 nn.ReLU(),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1))        #(-3+2+2)/2

#通道数增长率为32
num_channels,growth_rate=64,32
num_convs_in_dense_blocks=[4,4,4,4]

blks=[]

for i,num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs,num_channels,growth_rate))

    #对于1个稠密层来说，输出通道增加了 ：通道增长率*卷积块数
    num_channels+=num_convs*growth_rate

    #添加转换层使通道数减半,输出高、宽也减半
    if i!=len(num_convs_in_dense_blocks)-1:     #除了最后一个稠密层外
        blks.append(transition_block(num_channels,num_channels//2))
        #添加完记得更新通道数减半
        num_channels=num_channels//2

net=nn.Sequential(b1,
                  *blks,

                  #第一个稠密块输出通道：64+4*32，减半后96  第二个dense_block输出通道96+4*32，减半后112
                  #第3 输出112+4*32，减半120    第4输出120+4*32=248=num_channels
                  nn.BatchNorm2d(num_channels),

                  #全局平均池化
                  nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten(),
                  nn.Linear(num_channels,10))


if __name__=='__main__':
    # #查看各层输出形状
    # X=torch.rand(1,1,96,96)
    # for layer in net:
    #     X=layer(X)
    #     print(layer.__class__.__name__,"output shape：",X.shape)

    lr,num_epochs,batch_size=0.1,10,256
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=96)
    train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
    d2l.plt.show()













    '''
    #卷积块数2，输入通道3，输出通道10
    blk=DenseBlock(2,3,10)
    X=torch.rand(4,3,8,8)
    Y=blk(X)
    print(Y.shape)
    #torch.Size([4, 23, 8, 8])
    #第一个卷积块输入通道数是：10*0+3 ，输出通道10；   将输入和输出在通道维度连结后作为下一个卷积块的输入通道：3+10
    #第二个卷积块输入通道：10*1+3   ,输出通道：10     将输入和输出在通道维度连结后作为输出通道：13+10
    #对于输入通道数3 增加了卷积块数*通道增长率10

    #接入过渡层
    blk=transition_block(23,10)
    Z=blk(Y)
    print(Z.shape)
    torch.Size([4, 10, 4, 4])
    '''

