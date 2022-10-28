import numpy as np

#sigmod函数：变化方向不变，靠近0的位置，变化趋势较大，适用于标准化后的数据
def sigmod(x):
    y=1/(1+np.exp(-x))
    return y

x=np.array([0.2,0.3])#模拟输入层
b=np.array([0.6,0.3,0.5])#模拟偏置量
w=np.array([[0.7,0.3,0.5],[0.6,0.2,0.4]])#模拟权重

a=np.dot(x,w)+b#模拟神经元计算

z=sigmod(a)#模拟激活函数


print(a)
print(z)

#以上为模拟多层神经网络的中的一个神经元激活的过程，其他大致如此
#最后的输出层，由于目标不同，一般来说：回归问题使用恒等函数（直接输出），
#二元分类使用sigmoid函数（与0-1真实标签配合使用），多元分类使用softmax函数（与one-hot的真实标签配合使用）