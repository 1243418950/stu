import numpy as np
a = np.array([[2,4,5],[2,3,9]])
print(a.ndim) #维度 将会输出2
print(a.shape) #形状 将会输出（2，3）

b = np.array([[2,2,1],[2,2,4],[5,4,4]])

c=np.dot(a,b)

print(c)

