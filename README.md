# 南瓜品种分类与分析项目 
## 1. 项目概述
本项目基于美国南瓜（US-Pumpkins）数据集，利用机器学习技术对南瓜品种进行分类预测，并分析影响南瓜品种的关键特征。项目旨在通过随机森林算法，根据南瓜的产地、包装、大小等特征，准确预测其品种类型，为南瓜种植、销售和市场分析提供数据支持。

## 2. 数据准备与预处理
### 2.1 数据集特征
数据来源：US-pumpkins.csv
主要特征：
City Name：南瓜销售城市
Package：南瓜包装类型
Origin：南瓜产地
Item Size：南瓜大小
Low Price/High Price：南瓜价格区间
Mostly Low/Mostly High：价格主要分布区间
目标变量：Variety（南瓜品种）
### 2.2 数据预处理
缺失值处理：对分类变量使用众数填充，数值变量使用中位数填充
编码处理：使用LabelEncoder对分类变量进行编码转换
数据划分：按7:3比例划分训练集和测试集

## 3. 模型构建与评估
### 3.1 模型选择
采用随机森林分类器（RandomForestClassifier），参数设置：

n_estimators=100（树的数量）
random_state=42（随机种子）
### 3.2 模型评估
分类报告：
![image](https://github.com/user-attachments/assets/bb53d84b-b077-4909-aa2c-7bbfdb7a38e5)

## 4. 可视化分析
### 4.1 南瓜品种分布
![image](https://github.com/user-attachments/assets/5799706d-fd72-4271-a40d-5a68bddad017)

- 展示最常见的10种南瓜品种及其数量分布
- 横轴：南瓜品种（解码后的实际名称）
- 纵轴：数量
 
### 4.2 特征重要性分析
 
- 显示各特征对南瓜品种预测的重要性排序
- 关键特征包括：产地(Origin)、大小(Item Size)、包装(Package)等
- 横轴：特征名称
- 纵轴：重要性得分
