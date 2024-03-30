import autograd.numpy as np
from autograd import grad

# 定义目标函数
def objective_function(X):
    # 这里应该填入具体的损失函数计算逻辑
    return np.sum(X**2)

# 假设X是我们的初始参数矩阵，其中包含固定和可变元素
X = np.array([1.0, 2.0, 3.0, 4.0])

# 计算目标函数的梯度
gradient_function = grad(objective_function)

# 计算梯度
dX = gradient_function(X)

# 定义固定元素的索引
fixed_indices = [0, 2]  # 假设第一个和第三个元素是固定的

# 创建一个与梯度形状相同的数组，并初始化为0
dX_modified = np.zeros_like(dX)

# 将非固定元素的梯度复制到dX_modified中
for idx in range(len(X)):
    if idx not in fixed_indices:
        dX_modified[idx] = dX[idx]
dX = dX_modified
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.001
t = 0  # 迭代计数器
num_iterations = 100

m = np.zeros_like(X)  # 一阶矩估计
v = np.zeros_like(X)  # 二阶矩估计

for iteration in range(num_iterations):
    t += 1
    
    # 更新一阶矩和二阶矩估计
    m = beta1 * m + (1 - beta1) * dX
    v = beta2 * v + (1 - beta2) * (dX ** 2)
    
    # 计算偏差校正后的一阶矩和二阶矩估计
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # 更新参数
    X -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    # 计算新的梯度
    dX = gradient_function(X)
    dX_modified = np.zeros_like(dX)
    # 将非固定元素的梯度复制到dX_modified中
    for idx in range(len(X)):
        if idx not in fixed_indices:
            dX_modified[idx] = dX[idx]
    dX = dX_modified

