# -*- coding: utf-8 -*-
"""
python implementation of Genetic Algorithm
@author: Yifei Wang
"""

#%%
import numpy as np

#%% 参数设定
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

distance=getdistmat(position)

T = 1 #迭代次数
prob_1 = 0.2 #交叉概率
prob_2 = 0.2 #变异概率
M=10 #个体数
kill_num=2 #每次清除的个体数

#%% 函数定义

#计算适应度
def loss_function(ans):
    assert(ans.shape[0] == M)
    loss=np.zeros(shape=(M,1))
    for num,item in enumerate(ans):
        temp_loss=0
        for i in range(len(item)-1):
            temp_loss+=distance[i][i+1]
        temp_loss+=distance[0][len(item)-1]
        loss[num][0]=temp_loss
    return loss

#选择
def select(ans):
    assert(ans.shape[0] == M)
    loss=loss_function(ans)
    prob_kill=loss/np.sum(loss)
    prob_kill=prob_kill[:,0]
    kill=[]
    for i in range(kill_num):    
        while 1:
            temp=np.random.choice(M,1,p=prob_kill)[0]
            if not (temp in kill):
                kill.append(temp)
                break
    #print(kill)
    ans=np.delete(ans,obj=kill,axis=0)
    return ans

#交叉
def cross(ans):
    assert(ans.shape[0] == M-kill_num)
    for _ in range(kill_num):
        ans_new=np.zeros(shape=(1,distance.shape[0]))
        #随机选出亲代
        rand1=np.random.randint(low=0,high=len(ans)-1)
        rand2=rand1
        while 1: 
            rand2=np.random.randint(low=0,high=len(ans)-1)
            if not rand1 == rand2: #确保两个随机数不同
                break
        #随机选择的亲代
        f1=ans[rand1]
        f2=ans[rand2]
        #随机选择交叉区间，rand1、rand2为区间端点
        rand1=np.random.randint(low=0,high=distance.shape[0]-2)
        rand2=rand1
        rand2=np.random.randint(low=rand1+1,high=distance.shape[0])
        ans_new[0,rand1:rand2]=f1[rand1:rand2]
        temp=[]
        for i in f2:
            if not i in ans_new[0]:
                temp.append(i)
        j=0
        for i in range(0,rand1):
            ans_new[0,i]=temp[j]
            j+=1
        for i in range(rand2,len(ans_new[0])-1):
            ans_new[0,i]=temp[j]
            j+=1
        ans=np.vstack((ans,ans_new)) #将新生成的子代加入序列
    return ans

#变异
def variation(ans):
    choice_num=np.random.choice(M,int(prob_2*10))
    choice=ans[choice_num]
    for i,_ in enumerate(choice):
        rand1=np.random.randint(low=1,high=len(ans))
        while 1:
            rand2=np.random.randint(low=1,high=len(ans))
            if not rand1 == rand2: #确保两个随机数不同
                break
        temp=ans[i][rand1]
        ans[i][rand1]=ans[i][rand2]
        ans[i][rand2]=temp    
    return ans

#算法实现
def GA():
    print('start')
    ans=np.arange(0,distance.shape[0])
    for _ in range(M-1): #确定初始状态
        ans_temp=np.arange(0,distance.shape[0])
        ans=np.vstack((ans,ans_temp))
    #print(ans.shape)
    #print(ans)
    for _ in range(T): #迭代
        ans=select(ans)
        ans=cross(ans)
        ans=variation(ans)
        loss=loss_function(ans)
        #print(ans)
    return ans,loss


#%% 测试算法
if __name__ == "__main__":
    #选择最后一轮中最优的策略
    ans,loss=GA() 
    best_index=np.argmax(loss)
    best_ans=ans[best_index]
    best_loss=loss[best_index]
    '''
    print(best_ans)
    print('---------------')
    print(best_loss)
    print('---------------')
    '''