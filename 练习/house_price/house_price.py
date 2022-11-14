import numpy as np
import pandas
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


#加载训练集和测试集
train_data=pd.read_csv('../data/house-price/train.csv')
test_data=pd.read_csv('../data/house-price/test.csv')\

# print(train_data.shape)
# print(test_data.shape)
#
# print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])

#这里是获取全部样本的特征。去除第一维（第一列）所表示的序号,另外训练集需要额外去除最后一列数据：标签
all_feature=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

print(all_feature.shape)

numeric_features=all_feature.dtypes[all_feature.dtypes!='object'].index

#将全部数据标准化均值为0，方差为1
all_feature[numeric_features]=all_feature[numeric_features].apply(lambda x:(x-x.mean()/x.std()))

#标准化后可以将缺失值赋为0
all_feature[numeric_features] = all_feature[numeric_features].fillna(0)

#处理离散值，使用one_hot编码替代
#“MSZoning”的原始值为“RL”，则：“MSZoning_RL”为1，“MSZoning_RM”为0。
all_feature=pandas.get_dummies(all_feature,dummy_na=True)

# print(all_feature.shape)
n_train=train_data.shape[0]
train_features=torch.tensor(all_feature[:n_train].values,dtype=torch.float32)
test_features=torch.tensor(all_feature[n_train:].values,dtype=torch.float32)

#转换为列向量
train_labels=torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)

# print(train_features.shape," ",test_features.shape," ",train_labels.shape)

#定义损失
loss=nn.MSELoss()
in_features=train_features.shape[1]

#定义模型（线性）
def get_net():
    #线性模型，输出为1即预测房价
    net=nn.Sequential(nn.Linear(in_features,1))
    return net

#转换为对数，使用均方根误差
def log_rmse(net,features,labels):
# 为了在取对数时进⼀步稳定该值，将⼩于1的值设置为1
    clipped_preds=torch.clamp(net(features),1,float('inf'))
    rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(train_labels)))

    #返回迭代器
    return rmse.item()



#训练
def train(net,train_features,train_labels,test_features,test_labels,num_epochs,lr,wight_decay,batch_szie):

    train_ls,test_ls=[],[]
    #每次加载batch_szie批量大小的数据
    train_iter=d2l.load_array((train_features,train_labels),batch_szie)

    #优化器
    optimizer=torch.optim.Adam(net.parameters(),
                               lr=lr,
                               weight_decay=wight_decay)

    for epoch in range(num_epochs):
        for x,y in train_iter:      #获取训练特征和训练标签
            optimizer.zero_grad()   #梯度清零
            l=loss(net(x),y)
            l.backward()            #反向传播
            optimizer.step()        #更新参数
        train_ls.append(log_rmse(net,train_features,train_labels))

        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))

    return train_ls,test_ls




#获取每次验证的数据
def get_k_fold_data(k,i,x,y):
    assert k>1
    fold_size=x.shape[0]//k         #获取每个子集的大小。样本数除以k，向下取整
    x_train,y_train=None,None
    for j in range(k):      #

        #每次获取fold_size大小（每个子集大小）的数据集和对应大小的标签
        idx=slice(j * fold_size, (j + 1) * fold_size)
        x_part,y_part=x[idx,:],y[idx]

        #如果子集的序号和进行的第k次验证的次数相同则 保存该子集的数据和对应标签为验证集
        if j==i:
            x_valid,y_valid=x_part,y_part

        elif x_train is None:   #将第一个不符合上面条件的子集保存为训练集
            x_train,y_train=x_part,y_part
        else:
            x_train=torch.cat([x_train,x_part],0)   #其他的训练集连结在一起
            y_train=torch.cat([y_train,y_part],0)


    #返回训练集和测试集
    return x_train,y_train,x_valid,y_valid



#k折交叉验证

#k次训练，返回训练和验证误差的平均值
def k_fold(k,x_train,y_train,num_epochs,lr,weighr_decay,batch_size):
    train_l_sum,vaild_l_sum=0,0

    #进行k次验证
    for i in range(k):

        #每次将数据集分割为k个子集，每次在k-1个上训练，1个上测试
        #获取训练数据，训练标签，测试数据，测试标签
        data=get_k_fold_data(k,i,x_train,y_train)
        #模型
        net=get_net()
        #进行训练
        train_ls,valid_ls=train(net,*data,num_epochs,lr,weighr_decay,batch_size)


        #计算总的损失
        train_l_sum+=train_ls[-1]
        vaild_l_sum+=valid_ls[-1]

        #画图
        if i ==0:
            d2l.plot(list(range(1,num_epochs+1)),[train_ls,valid_ls],
                     xlabel='epoch',ylabel='rmse',xlim=[1,num_epochs],
                     legend=['train','valid'],yscale='log')

        print(f'折{i+1},训练log{float(train_ls[-1]):f},',f'验证log rmse{float(valid_ls[-1]):f}')

    return train_l_sum/k,vaild_l_sum/k


if  __name__=='__main__':
    k,num_epochs,lr,weight_decay,batch_size=5,100,5,0,64
    train_l,valid_l=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')
