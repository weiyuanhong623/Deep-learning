import os
import torch
import torchvision
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

# d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
# '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
# voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

#ImageSets/Segmentation路径包含⽤于训练和测试样本的⽂本⽂件，
# JPEGImages和SegmentationClass路径分别存储着每个⽰例的输⼊图像和标签。此处的标签也采⽤图像格式，其尺⼨和它所标注的输⼊图像的尺⼨。此外，标签中颜⾊相同的像素属于同⼀个语义类别。


voc_dir='../data/VOCdevkit/VOC2012/'



def read_voc_images(voc_dir,is_train=True):

    #读取所有voc图像并标注

    #读取voc图片的文本信息，这里是获取完整路径
    txt_fname=os.path.join(voc_dir,'ImageSets','Segmentation','train.txt' if is_train else 'val.txt')

    # 设置模式为：将JPEG或PNG图像读入三维RGB张量。
    mode=torchvision.io.image.ImageReadMode.RGB

    with open(txt_fname,'r') as f:      #获取句柄
        images=f.read().split()
    # images共1466个训练样本list类型

    features,labels=[],[]

    for i,fname in enumerate(images):

        #数据集
        features.append(torchvision.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{fname}.jpg')))

        #标签,读取对应voc图像时设置模式
        labels.append(torchvision.io.read_image(os.path.join(voc_dir,'SegmentationClass',f'{fname}.png'),mode))

    return features,labels

train_features,train_labels=read_voc_images(voc_dir,True)


#查看前5个样本和其标签
# n=5
# images=train_features[0:n]+train_labels[0:n]
# images=[img.permute(1,2,0) for img in images]
# d2l.show_images(images,2,n)     #图像将以2行5列的形式展示
# d2l.plt.show()

# 列举RGB颜⾊值和类名。
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 通 过 上 ⾯ 定 义 的 两 个 常 量， 我 们 可 以 ⽅ 便 地 查 找 标 签 中 每 个 像 素 的 类 索 引。 我 们 定 义
# 了voc_colormap2label函数来构建从上述RGB颜⾊值到类别索引的映射，⽽voc_label_indices函数
# 将RGB值映射到在Pascal VOC2012数据集中的类别索引。



def voc_colormap2label():
    #从RGB数值到voc类别索引的映射

    colormap2label=torch.zeros(256**3,dtype=torch.long)     #创建一维向量
    for i ,colormap in enumerate(VOC_COLORMAP):             #遍历RGB颜色值和类别映射表
        # print('具体是：',(colormap[0]*256+colormap[1])*256+colormap[2])
        colormap2label[ (colormap[0]*256+colormap[1])*256+colormap[2] ]=i     #一共21种类别
    return colormap2label           #存储类别的向量，转换后的数字为下标，值就是0-20 21个数字，其他值为0


def voc_label_indices(colomap,colormap2label):      #colomap是标签对应的voc图
    colomap=colomap.permute(1,2,0).numpy().astype('int32')          #把通道维调到最后,并由tensor转化为ndarray，还有数据类型(之前是dtype=torch.uint8))
    idx=( (colomap[:,:,0]*256+colomap[:,:,1])+colomap[:,:,2] )
    # print("idx",idx.shape)
    # print(idx[105:115, 130:140])
    return colormap2label[idx]          #返回对应类别标签


#使用第一个样本的voc图像标签做示例
# print(type(train_labels[0]))
# print(train_labels[0].shape)
# <class 'torch.Tensor'>
# torch.Size([3, 281, 500])

# col=train_labels[0].permute(1,2,0).numpy().astype('int32')
# print(col[2].shape)

# y=voc_label_indices(train_labels[0],voc_colormap2label())
# print(y[105:115, 130:140])



#F.crop()的接口就是F.crop(img, top, left, height, width)，参数分别是图片、H维坐标、W维坐标、裁剪的高、裁剪的宽

#输入图像预处理（进行随机裁切）
def voc_rand_crop(feature, label, height, width):
    # """随机裁剪特征和标签图像"""

    rect = torchvision.transforms.RandomCrop.get_params(            #设置参数
        feature, (height, width))

    feature = torchvision.transforms.functional.crop(feature, *rect)    #自定义的transform  ,同时输入图像和标签都进行裁切
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


#展示5个样本和对应标签
# n=5
# imgs=[]
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
#
#
# imgs = [img.permute(1, 2, 0) for img in imgs]           #把通道维放到最后
# d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
# d2l.plt.show()


#自定义语义分割数据集类
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir):

        self.transform = torchvision.transforms.Normalize(          #对图像RGB三个通道分别进行标准化
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.crop_size = crop_size      #定义尺寸大小
        features, labels = read_voc_images(voc_dir, is_train=is_train)  #读取数据集和标签

        self.features = [self.normalize_image(feature)          #对于数据集中有些图像的尺⼨可能⼩于随机裁剪所指定的输出尺⼨，这些样本可以通过⾃定义的filter函数移除掉
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)

        self.colormap2label = voc_colormap2label()      #存储类别的向量
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):     #像素值除以255后标准化
        return self.transform(img.float() / 255)

    def filter(self, imgs):             #定义过滤器
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):         #根据下标，随机裁剪
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                        *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


#读取数据集
# crop_size = (320, 480)      #限定图像大小
#
# voc_train = VOCSegDataset(True, crop_size, voc_dir)     #train 训练集和标签
# voc_test = VOCSegDataset(False, crop_size, voc_dir)     #test


# batch_size = 64
# train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
#         drop_last=True,
#         num_workers=3)


def load_data_voc(batch_size, crop_size):
    # """加载VOC语义分割数据集"""
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir),
                                             batch_size,shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir),
                                            batch_size,drop_last=True, num_workers=num_workers)
    return train_iter, test_iter



#net
#CNN部分——抽取图像特征

#残差块
class Residual(nn.Module):
    #输入通道数，输出通道数
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()

        #卷积层

        #第一层卷积层
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)

        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)

        if use_1x1conv:
            #即对于x使用1x1卷积层
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None

        #归一化层
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

    #定义正向传播                                 #整体是在ReLU激活函数之前进行残差映射或者恒等映射
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))       #先经过第一层卷积层然后进行归一化，接着进行ReLU激活函数
        Y=self.bn2(self.conv2(Y))               #接着进入第二层卷积层然后归一化
        if self.conv3:                          #对于x进行1x1卷积层或者不进行任何操作。
            X=self.conv3(X)
        Y+=X
        return  F.relu(Y)                       #映射后进行激活


def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk=[]

    #残差块数目
    for i in range(num_residuals):

        #对第一个残差块的x使用1x1卷积层
        if i ==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))

        #其他残差块的x不做变换
        else:
            blk.append(Residual(num_channels,num_channels))

    return blk

#输入通道是3
b1=nn.Sequential(nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

b2=nn.Sequential(*resnet_block(64,64,2,first_block=True))       #第一个模块中的第一个残差块的x不使用1x1卷积层
b3=nn.Sequential(*resnet_block(64,128,2))                       #使用1x1卷积层
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))

net=nn.Sequential(b1,b2,b3,b4,b5,           #去掉了全局平均汇聚、和输出类别的全连接
                  # nn.AdaptiveAvgPool2d((1,1)),
                  # nn.Flatten(),
                  # nn.Linear(512,10)
                  )

X=torch.rand(size=(1,3,320,480))
Z1=net(X).shape         #由这里可得高宽相比原高宽减少了32倍数
print(Z1)

#FCN部分
num_class=21        #类别是21
net.add_module('final_conv',nn.Conv2d(512,num_class,kernel_size=1))     #1*1卷积层使用⽤Xavier初始化
net.add_module('transpose_conv',nn.ConvTranspose2d(num_class,num_class,kernel_size=64,padding=16,stride=32))        #因为经过CNN部分输出的高宽减少了32倍，所以先确定stride为32，然后要达到成倍增加的效果要使得k=2p+s=


# Z=net(X).shape
# print(Z)

#双线性插值
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
        torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) *(1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                            kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

W=bilinear_kernel(num_class,num_class,64)       #⽤双线性插值的上采样初始化转置卷积层。
net.transpose_conv.weight.data.copy_(W)

# Z2=net(X).shape
# print(Z2)


def accuracy(y_hat, y):

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

def evaluate_accuracy_gpu(net,data_iter,device=None):
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
            metric.add(accuracy(net(X),y),y.numel())

    return metric[0]/metric[1]


def train_batch(net, X, y, loss, trainer, devices):

    if isinstance(X, list):     #对于list依次添加到显存
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])

    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, test_iter, loss, trainer, num_epochs,devices):

    timer, num_batches,timer2= d2l.Timer(), len(train_iter),d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    timer2.stop()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    timer2.stop()
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    print("total time: ",timer2.times[0]/60," min")





if __name__=='__main__':
# # print("finish")
# # #打印一个批量,
#     for x,y in train_iter:
#         print(x.shape)          #标签是⼀个三维数组。
#
#         print(y.shape)
#         break
    batch_size = 32
    crop_size = (320, 480)      #限定图像大小

    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
    train_iter, test_iter =load_data_voc(batch_size, crop_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    #训练
    def loss(inputs,targets):
        return F.cross_entropy(inputs,targets,reduction='none').mean(1).mean(1)

    train(net,train_iter,test_iter,loss,trainer,num_epochs,devices)
    # d2l.train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices)
    d2l.plt.show()





