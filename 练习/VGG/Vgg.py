import torch
from torch import nn
from d2l import torch as d2l




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
            l=loss(y_hat,y)
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

    print("total time:",timer1.times,"s")





#单个Vgg块
def vgg_block(num_convs,in_channels,out_channels):
    layers=[]

    #卷积部分
    for _ in range(num_convs):
        #卷积层；卷积核高、宽：3  上下各填充1，ph为2  所以输出和输入的形状相同，保持分辨率不变
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))

        #激活函数
        layers.append(nn.ReLU())

        #该层的输出通道数变成下一层卷积层的输入通道数
        in_channels=out_channels

    #池化部分       高、宽各减少2倍，总的减少4倍
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)



def vgg(conv_arch):
    conv_blks=[]
    in_channels=1

    for (num_convs,out_channels) in conv_arch:
        #卷积部分
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels=out_channels

    return nn.Sequential(*conv_blks,
                         nn.Flatten(),
                         nn.Linear(out_channels*7*7,4096),  #高宽需要除以2的五次方
                         nn.ReLU(),
                         nn.Dropout(p=0.5),

                         nn.Linear(4096,4096),
                         nn.ReLU(),
                         nn.Dropout(p=0.5),

                         nn.Linear(4096,10)
                         )


if __name__=='__main__':

    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]   #输出通道数变成4分之一

    net=vgg(small_conv_arch)



    # X=torch.rand(size=(1,1,224,224),dtype=torch.float32)
    # for blk in net:
    #     X=blk(X)
    #     print(blk.__class__.__name__,"output shape\t",X.shape)


    #训练
    lr,num_epochs,batch_size=0.05,10,128
    #读取数据
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
    d2l.plt.show()