#切不可在权重初始化时将初始值设为0，因为如果这样，
# 在误差反向传播中，所有的权重值的更新都将一致，使得权重归一化
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): #sigmoid激活函数
    return 1/(1+np.exp(-x))
#ReLU函数：自变量小于0时，输出0；自变量大于0时，输出为输入值
def ReLU(x):
    y=np.maximum(0,x)
    return y
x=np.random.randn(1000,100) #1000个数据

node_num=100 #各隐藏层的节点（神经元）数
hidden_layer_size=5 #5层神经元
activations={} #激活值的结果保存在这里

for i in range(hidden_layer_size):
    if i!=0:
        x=activations[i-1] #上一层的激活函数输出数据
    #w=np.random.randn(node_num,node_num)*1 #使用标准差为1的高斯分布作为权重初始值,激活值聚集在0-1两边，导致梯度消失现象
    #w = np.random.randn(node_num, node_num) * 0.01 #激活值聚集在0.5附近，导致激活值相同了，深层神经网络无意义了
    # w = np.random.randn(node_num, node_num) /np.sqrt(node_num) #sigmoid、tanh等线性激活函数适用于Xavier初始值
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num/2)  # ReLU非线性激活函数适用于He初始值
    z=np.dot(x,w)
    a=ReLU(z)
    activations[i]=a

for i,a in activations.items():
    #print(i,a)
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()
