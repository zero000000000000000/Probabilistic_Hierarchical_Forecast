# Week3

## DeepAR

1. 使用相关时间序列的数据（其他家庭的能源消耗、对其他产品的需求）不仅可以在不过度拟合的情况下拟合更复杂（因此可能更准确）的模型，还可以减轻选择和准备协变量以及选择经典技术所需模型的人工和劳动密集步骤
2. 在实际的预测问题中，特别是在需求预测领域，人们经常面临高度块状或间歇性的数据，这些数据违反了许多经典技术的核心假设，如时间序列的高斯性、平稳性或同方差性。长期以来，人们一直认为这是一个重要问题。
3. 深度神经网络为这种管道提供了一种替代方案。这样的模型需要有限数量的标准化数据预处理，之后通过学习端到端模型来解决预测问题。特别是，数据处理被包括在模型中，并共同优化，以实现产生尽可能好的预测的目标。在实践中，深度学习预测管道几乎完全依赖于模型可以从数据中学习到的东西，而传统管道在很大程度上依赖于启发式方法，如专家设计的组件和手动协变量设计



## 模拟

1. 离散：VAR（1） 二项式实现  合理系数取值保证平稳 相关为正
2. ARIMA，p q随机 参数空间保证平稳+可逆 相关为正
3. 