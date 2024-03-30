from autograd import grad
import autograd.numpy as np

# 定义一个接受矩阵作为输入的函数
def my_function(X):
    # 这里是一个简单的示例，如矩阵的Frobenius范数
    return np.square(np.linalg.norm(X))

# 创建一个矩阵
X = np.array([[1, 2], [3, 4]])

# 计算函数的梯度
gradient_function = grad(my_function)

# 计算在矩阵X处的梯度
gradient = gradient_function(X)

print("矩阵 X:")
print(X)
print("函数 my_function(X) 的梯度:")
print(gradient)