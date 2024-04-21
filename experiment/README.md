# 实证实验

此文件夹主要展示实证实验的代码、图片等资源。

本次实证实验将聚焦于Tourism和Wiki两个公开数据集。

## Explanation

以下是文件夹及源代码文件的描述，未描述的均为弃用的文件

* `Data`:  原始数据、预处理数据、加和矩阵S
* `EDA`: 探索性数据分析，主要进行数据可视化与预处理
* `Base_Forecasts`: 基础预测
  * `base_forecasts.py`: 测试集的基础预测
  * `base_forecasts_in.py`: 验证集的基础预测
  * `DeepAR_forecast.py`: DeepAR生成的测试集基础预测
  * `DeepAR_preprocess.py`: 匹配DeepAR输入格式的预处理
* `Reconcile_and_Evaluation`: 
  * `opt_with_adam_regularization.py`: 正则化的adam优化
  * `opt_with_adam_regularization_optuna.py`: 正则化的adam优化超参数寻优
  * `produce_tourism_s.r`: 生成转化后的S矩阵
  * `r_tourism_new.r`: 评估ETS基础预测的所有方法
  * `tourism_deepar.r`: 评估DeepAR基础预测的所有方法

## To Run

