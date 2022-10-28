import numpy as np
import matplotlib.pyplot as plt

#阶跃函数：自变量大于0，输出1；自变量小于0，输出0.
def step_function(x):
    y=x>0
    y=y.astype(int)
    return y

#sigmod函数：变化方向不变，靠近0的位置，变化趋势较大，适用于标准化后的数据
def sigmod(x):
    y=1/(1+np.exp(-x))
    return y

#ReLU函数：自变量小于0时，输出0；自变量大于0时，输出为输入值
def ReLU(x):
    y=np.maximum(0,x)
    return y

if __name__ == '__main__':
     x1 = np.arange(-6, 7, 1)
     y1 = step_function(x1)
     #y1=sigmod(x1)
     #y1=ReLU(x1)
     plt.plot(x1, y1)
     plt.show()









