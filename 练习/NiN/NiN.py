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

    print("total time: ",timer1.times/60," min")

#单个NiN块
def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(#
                         nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                         nn.ReLU(),

                         #核为1*1的卷积层
                         nn.Conv2d(out_channels,out_channels,kernel_size=1),
                         nn.ReLU(),

                         #核为1*1的卷积层
                         nn.Conv2d(out_channels,out_channels,kernel_size=1),
                         nn.ReLU())




if __name__=='__main__':

    # 输入通道数、输出通道数、卷积核规格、步幅、填充
    # 使用fashion-mnist，扩充为224
    net = nn.Sequential(nin_block(1, 96, kernel_size=11, stride=4, padding=0),
                        # 由于只含一个卷积层，其他两个是1*1卷积层，这里只考虑第一层卷积层：输出的形状：高、宽（-11+4）//4  (224-11+4)//4=54
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        # 高、宽(-3+2)//2   26

                        nin_block(96, 256, kernel_size=5, stride=1, padding=2),     #padding=2 ph=pw=2*2=4
                        # 高、宽：-5+4+1   26
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        # 高、宽：12

                        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
                        # 高、宽 -3+2+1   12
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        #高、宽 5


                        nn.Dropout(p=0.5),

                        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
                        #高、宽：-3+2+1  5

                        # 输出的形状为1*1
                        nn.AdaptiveAvgPool2d((1, 1)),

                        # 展平
                        nn.Flatten()
                        )
    #查看网络结构
    X=torch.rand(size=(1,1,224,224))
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)


    lr,num_epochs,batch_sizze=0.1,10,128
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_sizze,resize=224)
    train_ch6(net,train_iter,test_iter,num_epochs,lr,device=d2l.try_gpu())
    d2l.plt.show()