import numpy as np

### 训练或者预测可能需要的参数

# 设置维度
d = 1

# 设置定义域
lb = np.array([0 for i in range(d)])
ub = np.array([1 for i in range(d)])


# 最小特征值真解
lambda_ =  d*np.pi**2


def Exact_u(x):
    u = 1
    for i in range(d):
        u = u * np.sin(np.pi*x[i])
    return u