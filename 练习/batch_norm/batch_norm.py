import torch
from torch import  nn
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


#moving_  是全局上的均值和方差
def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):

    #判断模式
    if not torch.is_grad_enabled():

        #预测模式（sqrt：逐元素求平方根）
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)        #对X进行偏移和缩放（使用全局均值和全局方差）
    else:
        #训练模式
        assert  len(X.shape) in (2,4)           #规格是2或者4维

        #全连接层（计算特征维上的均值和方差）
        if len(X.shape)==2:                     #如果是2维度，第一维度就是批量大小，第二维度就是特征
            mean=X.mean(dim=0)                  #求均值（按行对每列求均值，按照特征来求）
            var=((X-mean)**2).mean(dim=0)       #求方差

        #卷积层
        else:

            #使用二维卷积：计算通道维度（axis=1,:输入通道数）的均值和方差
            #保持X的形状以便后面可以做广播运算
            mean=X.mean(dim=(0,2,3),keepdim=True)       #计算通道维度的每个批量的高、宽（1*n*1*1# ）
            var=((X-mean)**2).mean(dim=(0,2,3),keepdim=True)        #

        #训练模式下对X进行偏移和缩放
        X_hat=(X-mean)/torch.sqrt(var+eps)

        #更新全局的均值和方差
        moving_mean=momentum*moving_mean+(1.0-momentum)*mean
        moving_var=momentum*moving_var+(1.0-momentum)*var

    Y=gamma*X_hat+beta      #进行线性变换,两个参数需要迭代更新。
    return Y,moving_mean.data,moving_var.data


#规范化层
class BatchNorm(nn.Module):
    #num_features；表示完全连接层的输出数量或卷积层的输出通道数。
    #num_dims：2表示完全连接，4表示卷积层

    def __init__(self,num_features,num_dims):
        super().__init__()

        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)

        #参与梯度和迭代的拉伸和偏移参数，分别初始化1和0
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))

        #非模型参数的变量初始化为0和1
        self.moving_mean=torch.zeros(shape)
        self.moving_var=torch.ones(shape)


    #定义前向计算
    def forward(self,X):
        # 如果X不在内存上，将moving_mean和moving_var 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
        self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y


net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5),
                  BatchNorm(6,num_dims=4),
                  nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),      #高宽减半

                  nn.Conv2d(6,16,kernel_size=5),
                  BatchNorm(16,num_dims=4),
                  nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2,stride=2),      #高宽减半

                  nn.Flatten(),
                  nn.Linear(16*4*4,120),
                  BatchNorm(120,num_dims=2),
                  nn.Sigmoid(),

                  nn.Linear(120,84),
                  BatchNorm(84,num_dims=2),
                  nn.Sigmoid(),

                  nn.Linear(84,10)

                  )


if __name__=='__main__':
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
    d2l.plt.show()