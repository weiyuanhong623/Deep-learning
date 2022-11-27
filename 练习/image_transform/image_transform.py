import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib as plt


#设置显示svg图片并设置图片大小
d2l.set_figsize()
img=d2l.Image.open('../data/images/aqua_1.jpg')
d2l.plt.imshow(img)
d2l.plt.show()      #为了显示图像


#aup：各种transform方法
#num_rows=2, num_cols=4 进行了2行4列共8次变换，并输出2行4列格式的图像
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]      #主要是进行2*4次变换
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

#0.5概率左右翻转
# apply(img,torchvision.transforms.RandomHorizontalFlip())

#上下翻转
# apply(img,torchvision.transforms.RandomVerticalFlip())

#随机裁剪面积为原始面积10%到100%的区域，该区域的宽高比从0.5到2之间随机取值，然后区域的宽度和⾼度都被缩放到200像素。
#a和b之间的随机数指的是在区间[a, b]中通过均匀采样获得的连续值。

shape_aug=torchvision.transforms.RandomResizedCrop(
    (200,200),scale=(0.1,1),ratio=(0.5,2)
)
# apply(img,shape_aug)

#改变颜色
# 随机更改图像的亮度，随机值为原始图像的50%（1 − 0.5）到150%（1 + 0.5）之间。
# apply(img,torchvision.transforms.ColorJitter(
#     brightness=0.5,contrast=0,saturation=0,hue=0
# ))

#随机改变色调
#的亮度（brightness）、对⽐度（contrast）、饱和度（saturation）和⾊调（hue）。
# apply(img,torchvision.transforms.ColorJitter(
#     brightness=0,contrast=0,saturation=0,hue=0.5
# ))

colo_augs=torchvision.transforms.ColorJitter(
    brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5
)
#结合多种
augs=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                     colo_augs,shape_aug])

apply(img,augs)