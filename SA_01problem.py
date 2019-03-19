# -*- coding: utf-8 -*-
"""
python implementation of SA algorithm
@author: Yifei Wang
"""

import math
import numpy as np

#%% 物品参数设定

value=np.array([1,5,8,5,3,6,4,9,5,4,2,5]) #物品价值
weight=np.array([4,7,9,7,3,3,6,9,0,8,6,5]) #物品重量
num=len(value) #物品个数

#算法参数设定
init_T=100 #起始温度
end_T=2 #阈值温度
iter_num=100 #特定温度下迭代次数
space=30 #背包最大容量

#%% 函数定义

#定义损失函数,此处取价值之和的相反数
def loss_function(current_choice):
    assert(current_choice.shape[0] == num)
    return -np.sum(current_choice*weight)

#定义随机函数
def random_choice(ans):
    assert(ans.shape[0] == num)
    #随机翻转某一位,即拿或不拿该件物品
    while 1:
        temp=np.random.randint(0,num)
        if ans[temp] == 1:
            ans[temp]=0
        else:
            ans[temp]=1
        if np.sum(ans*weight) <= space:
            break
    return ans

#SA算法实现
def SA(value=[],weight=[]):
    assert(value.shape[0] ==  weight.shape[0])
    print('start')
    ans=np.zeros(shape=num) #初始条件设为全部不取
    T=init_T
    while T>=end_T: #降温至阈值之下之前
        #print('new temperature')
        for _ in range(iter_num): #迭代iter_num次
            loss=loss_function(ans) #当前loss
            ans_new=random_choice(ans) #随机翻转后ans
            loss_new=loss_function(ans_new) #随机翻转后loss
            if loss_new < loss: #如果翻转后loss更小则直接接受
                ans=ans_new
            else: #如果翻转后loss更大则以一定概率接受
                prob=math.exp(-(loss_new-loss)/T)
                rand=np.random.uniform(low=0,high=1)
                if rand>prob:
                    ans=ans_new
        T=init_T/(1+T) #降温
    
    return ans #返回解


#%% 测试算法

if __name__=="__main__":
    ans=SA(value=value,weight=weight)
    print('final choice:',ans)
    print('--------------------')
    print('final value:',int(-loss_function(ans)))
    print('--------------------')
    print('final weight:',np.sum(ans*weight))