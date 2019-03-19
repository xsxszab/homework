# -*- coding: utf-8 -*-
"""
python implementation of SA algorithm
@author: Yifei Wang
"""

#%%
import math
import numpy as np
import networkx as nx #图可视化

#%%

"""
地图参数和转换成距离矩阵我是直接复制粘贴的，自己编数据太麻烦了......
"""
position = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],
                        [880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],
                        [1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],
                        [725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],
                        [300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],
                        [1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],
                        [420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],
                        [685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],
                        [475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],
                        [830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],
                        [1340.0,725.0],[1740.0,245.0]])


def getdistmat(coordinates):
    num = coordinates.shape[0] #52个坐标点
    distmat = np.zeros((52,52)) #52X52距离矩阵
    for i in range(num):
        for j in range(i,num):
            distmat[i][j] = distmat[j][i]=np.linalg.norm(coordinates[i]-coordinates[j])
    return distmat

distance=np.array(getdistmat(position))

#算法参数设定
init_T=100 #起始温度
end_T=2 #阈值温度
iter_num=100 #特定温度下迭代次数

#%% 函数定义

#损失函数
def loss_function(ans):
    assert(ans.shape[0] == distance.shape[0])
    loss=0
    for i in range(len(ans)-1):
        loss+=distance[i][i+1]
    loss+=distance[0][len(ans)-1]
    return loss

#随机选择函数
def random_choice(ans):
    assert(ans.shape[0] == distance.shape[0])
    #随机交换两个节点的遍历顺序
    rand1=np.random.randint(low=1,high=len(ans))
    while 1: 
        rand2=np.random.randint(low=1,high=len(ans))
        if not rand1 == rand2: #确保两个随机数不同
            break
    temp=ans[rand1]
    ans[rand1]=ans[rand2]
    ans[rand2]=temp
    return ans
#SA算法实现
def SA():
    print('start')
    T=init_T
    ans=np.arange(0,distance.shape[0])
    while T>=end_T: #降温至阈值之下之前
        #print('new temperature')
        for _ in range(iter_num): #迭代iter_num次
            loss=loss_function(ans) #当前loss
            ans_new=random_choice(ans) #随机处理ans
            loss_new=loss_function(ans_new) #随机处理后的loss
            if loss_new < loss: #如果翻转后loss更小则直接接受
                ans=ans_new
            else: #如果翻转后loss更大则以一定概率接受
                prob=math.exp(-(loss_new-loss)/T)
                rand=np.random.uniform(low=0,high=1)
                if rand>prob:
                    ans=ans_new
        T=init_T/(1+T) #降温
    
    return ans #返回解

#%%

if __name__ == "__main__":
    ans=SA()
    print('final choice:',ans)
    print('------------------')
