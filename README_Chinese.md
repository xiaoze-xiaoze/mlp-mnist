# 使用PyTorch构建MNIST手写数字分类神经网络

## 项目简介

**🎯 本项目专为深度学习初学者设计，采用代码教学的方式，通过详细的原理解释和逐行代码分析，帮助您深入理解神经网络的工作原理。**

本教程将带您从零开始，使用PyTorch深度学习框架构建一个多层感知机（MLP）神经网络，用于识别MNIST数据集中的手写数字。我们会详细解释每一行代码的作用、每个概念的原理，以及为什么要这样设计。

MNIST数据集包含70,000张28×28像素的灰度手写数字图像（0-9），是计算机视觉领域的"Hello World"项目，非常适合作为理解神经网络的起点。

> **📚 系列说明**：这是深度学习新手向代码教学系列的第一个项目。后续我会陆续更新其他深度学习模型的详细教学，包括CNN、RNN、Transformer等，敬请期待！

## 目录

1. [项目概述](#项目概述)
2. [神经网络架构详解](#神经网络架构详解)
3. [数据加载与预处理](#数据加载与预处理)
4. [模型实现原理](#模型实现原理)
5. [训练过程详解](#训练过程详解)
6. [模型评估与可视化](#模型评估与可视化)
7. [结果分析](#结果分析)
8. [核心概念解释](#核心概念解释)
9. [代码逐行解析](#代码逐行解析)
10. [总结与扩展](#总结与扩展)

## 项目概述

我们的神经网络实现包含以下特性：
- **多层感知机（MLP）**：4层全连接神经网络
- **批处理训练**：提高训练效率
- **GPU加速支持**：自动检测并使用GPU
- **交叉熵损失函数**：适用于多分类任务
- **Adam优化器**：自适应学习率优化
- **实时可视化**：展示预测结果和训练曲线

## 神经网络架构详解

### 网络结构设计

我们的神经网络采用简单而有效的全连接架构：

```
输入层 (784个神经元) → 展平28×28图像
隐藏层1 (128个神经元) → 特征提取
隐藏层2 (128个神经元) → 特征组合
隐藏层3 (64个神经元) → 特征压缩
输出层 (10个神经元) → 分类结果（0-9数字）
```

### 架构设计原理

1. **输入展平**：将28×28的二维图像转换为784维的一维向量
2. **渐进式特征学习**：
   - 第一层（784→128）：从原始像素中提取基本特征
   - 第二层（128→128）：组合基本特征形成更复杂的模式
   - 第三层（128→64）：压缩特征，保留最重要的信息
3. **输出层**：10个神经元对应10个数字类别（0-9）

### 为什么选择这种架构？

- **全连接层**：每个神经元都与前一层的所有神经元相连，能够学习全局特征
- **逐层递减**：从128→128→64的设计有助于特征的逐步抽象和压缩
- **适中的深度**：4层网络既能学习复杂模式，又不会过于复杂导致训练困难

## 数据加载与预处理

### MNIST数据集特征

- **训练集**：60,000张图像
- **测试集**：10,000张图像
- **图像尺寸**：28×28像素（灰度图）
- **类别数量**：10个数字（0-9）
- **像素值范围**：0-255（预处理后归一化为0-1）

### 预处理流程

```python
transform=ToTensor()  # 将PIL图像转换为张量并归一化到[0,1]
```

`ToTensor()`变换的作用：
1. **格式转换**：将PIL图像或numpy数组转换为PyTorch张量
2. **数值归一化**：将像素值从[0, 255]缩放到[0.0, 1.0]
3. **维度调整**：将数据布局从HWC（高度×宽度×通道）改为CHW（通道×高度×宽度）

### 为什么需要归一化？

- **数值稳定性**：避免梯度爆炸或消失
- **训练效率**：相同量级的输入有助于优化器更好地工作
- **激活函数效果**：归一化的输入使激活函数工作在最佳区间

## 模型实现原理

### 类结构设计

我们的`NeuralNetwork`类继承自`nn.Module`，这是PyTorch中所有神经网络的基类：

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 定义网络层
        
    def forward(self, x):
        # 定义前向传播过程
```

### 层级详细解析

1. **nn.Flatten()**：
   - 功能：将输入从(batch_size, 1, 28, 28)重塑为(batch_size, 784)
   - 原理：保持批次维度不变，将图像数据展平为一维向量

2. **nn.Linear(784, 128)**：
   - 功能：第一个全连接层
   - 参数：权重矩阵W(784×128) + 偏置向量b(128)
   - 计算：output = input × W + b
   - 参数数量：784 × 128 + 128 = 100,480个

3. **后续全连接层**：
   - fc2: 128→128，参数数量：128 × 128 + 128 = 16,512个
   - fc3: 128→64，参数数量：128 × 64 + 64 = 8,256个
   - fc4: 64→10，参数数量：64 × 10 + 10 = 650个

4. **总参数数量**：约125,898个可训练参数

### 前向传播过程

```python
def forward(self, x):
    x = self.flatten(x)    # 展平：(batch, 1, 28, 28) → (batch, 784)
    x = self.fc1(x)        # 第一层：(batch, 784) → (batch, 128)
    x = self.fc2(x)        # 第二层：(batch, 128) → (batch, 128)
    x = self.fc3(x)        # 第三层：(batch, 128) → (batch, 64)
    x = self.fc4(x)        # 输出层：(batch, 64) → (batch, 10)
    return x
```

**注意**：本实现中没有使用激活函数，这是一个简化版本。在实际应用中，通常会在隐藏层之间添加ReLU等激活函数。

## 训练过程详解

### 训练循环的核心组件

#### 前向传播（Forward Pass）
1. **输入处理**：将一批图像输入网络
2. **预测生成**：网络输出每个类别的原始分数（logits）
3. **损失计算**：使用交叉熵损失比较预测值与真实标签

#### 反向传播（Backward Pass）
1. **梯度计算**：`loss.backward()`计算所有参数的梯度
2. **参数更新**：`optimizer.step()`根据梯度更新权重
3. **梯度清零**：`optimizer.zero_grad()`清除上一次的梯度

### 关键训练参数

- **批次大小（Batch Size）**：64
  - 平衡内存使用和训练稳定性
  - 较大批次：更稳定的梯度，更好的GPU利用率
  - 较小批次：更频繁的更新，更少的内存使用

- **学习率（Learning Rate）**：0.0045
  - 针对Adam优化器调优的学习率
  - 控制参数更新的步长

- **训练轮数（Epochs）**：25
  - 完整遍历训练数据的次数
  - 通常足够让模型在MNIST上收敛

- **优化器**：Adam
  - 自适应学习率算法
  - 结合了动量和自适应学习率的优势

### 训练过程监控

```python
if batch_size_num % 100 == 0:
    print(f"loss:{loss_value:>7f}    [number:{batch_size_num}]")
```

每100个批次打印一次损失值，帮助监控训练进度和收敛情况。

## 模型评估与可视化

### 评估指标

我们的测试函数提供全面的模型评估：

1. **准确率计算**：正确分类图像的百分比
2. **损失跟踪**：测试集上的平均交叉熵损失
3. **可视化检查**：随机显示10个测试样本及其预测结果

### 可视化功能

1. **样本预测展示**：
   - 显示原始图像
   - 对比真实标签和预测标签
   - 直观评估模型性能

2. **训练曲线**：
   - 绘制训练损失随轮数的变化
   - 帮助识别收敛和过拟合

3. **实时反馈**：
   - 每100个批次显示训练进度
   - 每轮训练后显示测试结果

## 结果分析

### 预期性能表现

使用本架构和超参数，您可以期待：
- **训练准确率**：~95-98%
- **测试准确率**：~95-97%
- **训练时间**：GPU上2-3分钟，CPU上10-15分钟
- **收敛速度**：通常在15-20轮内收敛

### 性能影响因素

1. **网络深度**：4层网络能够学习足够复杂的特征
2. **参数数量**：约12.6万参数足以处理MNIST的复杂度
3. **Adam优化器**：自适应学习率提高收敛速度
4. **批处理**：稳定的梯度估计

### 可能的改进方向

1. **添加激活函数**：在隐藏层间添加ReLU激活函数
2. **正则化技术**：添加Dropout防止过拟合
3. **批归一化**：加速训练并提高稳定性
4. **学习率调度**：动态调整学习率

## 核心概念解释

### 交叉熵损失（Cross-Entropy Loss）

交叉熵损失是多分类任务的理想选择：

```
Loss = -Σ(y_true * log(y_pred))
```

**优势**：
- 对错误的置信预测给予重罚
- 提供强梯度信号促进学习
- 与softmax输出配合良好

**工作原理**：
- 真实标签为one-hot编码
- 预测概率通过softmax计算
- 损失值越小表示预测越准确

### Adam优化器

Adam结合了AdaGrad和RMSprop的优势：

**核心特性**：
- **自适应学习率**：为每个参数设置不同的学习率
- **动量机制**：在一致方向上加速收敛
- **偏置修正**：修正初始化偏差

**数学原理**：
```
m_t = β₁ * m_{t-1} + (1-β₁) * g_t        # 动量
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²       # 二阶矩
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)    # 参数更新
```

**其他常见优化器对比**：
- **SGD（随机梯度下降）**：最基础的优化器，学习率固定
- **Momentum**：在SGD基础上加入动量，加速收敛
- **RMSprop**：自适应学习率，适合处理稀疏梯度
- **AdaGrad**：累积梯度平方，但学习率会持续衰减

**为什么选择Adam？**
- 结合了动量和自适应学习率的优势
- 对超参数不敏感，通常使用默认参数就有很好效果
- 在大多数深度学习任务中表现稳定
- 特别适合处理稀疏梯度和噪声数据

### 批处理（Batch Processing）

批处理提供多重优势：

1. **内存效率**：在有限内存中处理大数据集
2. **梯度稳定**：减少梯度估计的噪声
3. **并行化**：充分利用GPU/CPU的并行处理能力
4. **训练稳定性**：平滑的损失曲线

**批次大小选择原则**：
- 太小：训练不稳定，GPU利用率低
- 太大：内存不足，泛化能力差
- 64是一个经验上的良好平衡点

## 代码逐行解析

### 1. 导入库和依赖项

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
```

**详细说明**：
- `torch`：PyTorch核心库，提供张量操作和神经网络功能
- `torch.nn`：包含神经网络层、损失函数和工具
- `DataLoader`：处理批处理和数据迭代
- `torchvision.datasets`：提供常用数据集（如MNIST）的访问
- `ToTensor()`：将PIL图像转换为PyTorch张量的变换
- `matplotlib.pyplot`：用于可视化和绘图
- `numpy`：数值计算和数组处理

### 2. 数据加载和准备

```python
# 下载训练集
training_data = datasets.MNIST(
    root="data",             # 数据集存放的路径
    train=True,              # 是否为训练集
    download=True,           # 是否下载
    transform=ToTensor(),    # 数据转换
)

# 下载测试集
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

**逐行解析**：
- `root="data"`：在当前目录创建'data'文件夹存储数据集
- `train=True/False`：指定加载训练集（60,000张）还是测试集（10,000张）
- `download=True`：如果数据集不存在则自动下载
- `transform=ToTensor()`：应用预处理变换：
  - 将PIL图像（0-255）转换为PyTorch张量（0.0-1.0）
  - 改变格式从HWC（高度×宽度×通道）到CHW
  - 确保数据类型与神经网络兼容

### 3. 数据可视化函数

```python
def show_samples(dataset, model, n_samples=10):
    fig, axes = plt.subplots(1, n_samples, figsize=(15,3))
    indices = np.random.choice(len(dataset), n_samples)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred_label = pred.argmax().item()

        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"True: {label}\nPred: {pred_label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
```

**逐行分析**：
- **第2行**：创建子图网格（1行，n_samples列）
- **第3行**：从数据集中随机选择样本索引
- **第6行**：获取图像和真实标签
- **第7行**：`torch.no_grad()`禁用梯度计算（推理时节省内存）
- **第8行**：
  - `img.unsqueeze(0)`：添加批次维度：(28,28) → (1,28,28)
  - `.to(device)`：将张量移动到GPU/CPU
  - `model(...)`：执行前向传播
- **第9行**：`argmax()`找到概率最高的类别，`.item()`转换为Python整数
- **第11行**：`img.squeeze()`移除单一维度用于显示
- **第12行**：显示真实标签和预测标签

### 4. 创建数据加载器

```python
train_dataloader = DataLoader(training_data, batch_size=64)    # 训练集
test_dataloader = DataLoader(test_data, batch_size=64)         # 测试集

for x,y in train_dataloader:
    print(f"shape of x[N,C,H,W]:{x.shape}")    # 图像形状
    print(f"shape of y:{y.shape,y.dtype}")     # 标签形状和数据类型
    break
```

**解释**：
- **DataLoader**：从数据集创建可迭代的批次
- **batch_size=64**：同时处理64张图像
  - 更大批次：更稳定的梯度，更好的GPU利用率
  - 更小批次：更频繁的更新，更少的内存使用
- **形状分析**：
  - `x.shape`：(64, 1, 28, 28) = (批次大小, 通道数, 高度, 宽度)
  - `y.shape`：(64,) = 64个标签的批次
  - `y.dtype`：torch.int64（适用于分类任务）

### 5. 设备配置

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

**目的**：
- **GPU加速**：CUDA支持NVIDIA GPU上的并行处理
- **自动回退**：如果GPU不可用则使用CPU
- **性能影响**：GPU训练速度可比CPU快10-100倍

### 6. 神经网络架构实现

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
```

**详细架构分析**：

1. **类定义**：
   - 继承自`nn.Module`，PyTorch神经网络的基类
   - `super().__init__()`：调用父类构造函数

2. **层定义**：
   - `nn.Flatten()`：将2D图像转换为1D向量
   - `nn.Linear(784, 128)`：全连接层，784输入→128输出
   - 参数计算：784 × 128 + 128 = 100,480个参数

3. **前向传播**：
   - 数据依次通过每一层
   - 每层进行线性变换：output = input × weight + bias
   - 最终输出10个类别的原始分数（logits）

4. **参数统计**：
   - fc1: 784×128+128 = 100,480
   - fc2: 128×128+128 = 16,512
   - fc3: 128×64+64 = 8,256
   - fc4: 64×10+10 = 650
   - **总计**：125,898个可训练参数

### 7. 模型实例化

```python
model = NeuralNetwork().to(device)
print(model)
```

- 创建模型实例并移动到指定设备（GPU/CPU）
- 打印模型结构以验证架构

### 8. 训练函数详细实现

```python
def train(dataloader, model, loss_fn, optimizer):
    model.train()                    # 设置为训练模式
    epoch_loss = 0
    batch_size_num = 1

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)   # 将数据移动到GPU/CPU

        # 前向传播
        pred = model.forward(x)              # 获取预测结果
        loss = loss_fn(pred, y)              # 计算损失

        # 反向传播
        optimizer.zero_grad()                # 清除之前的梯度
        loss.backward()                      # 计算梯度
        optimizer.step()                     # 更新参数

        # 记录和跟踪
        loss_value = loss.item()
        epoch_loss += loss_value
        if batch_size_num % 100 == 0:
            print(f"loss:{loss_value:>7f}    [number:{batch_size_num}]")
        batch_size_num += 1

    # 计算该轮的平均损失
    avg_loss = epoch_loss / len(dataloader)
    train_loss_history.append(avg_loss)

    # 绘制训练loss曲线
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

    return avg_loss
```

**逐步训练过程解析**：

1. **model.train()**：启用训练模式
   - 激活Dropout（如果存在）
   - 启用批归一化的训练行为

2. **数据移动**：`x.to(device), y.to(device)`
   - 将张量传输到GPU以加速计算
   - 对GPU训练至关重要

3. **前向传播**：`pred = model.forward(x)`
   - 输入数据流经所有网络层
   - 产生每个类别的原始分数（logits）

4. **损失计算**：`loss = loss_fn(pred, y)`
   - 交叉熵损失比较预测与真实标签
   - 损失值越高表示性能越差

5. **梯度重置**：`optimizer.zero_grad()`
   - PyTorch默认累积梯度
   - 每次反向传播前必须清零

6. **反向传播**：`loss.backward()`
   - 使用链式法则计算梯度
   - 梯度存储在parameter.grad属性中

7. **参数更新**：`optimizer.step()`
   - 使用计算的梯度更新权重
   - Adam优化器应用自适应学习率

8. **进度跟踪**：
   - 每100个批次记录损失
   - 计算轮次平均损失
   - 存储历史记录用于绘图

9. **实时可视化**：
   - 每个epoch结束后立即绘制训练曲线
   - 实时监控训练进度和收敛情况
   - 帮助及时发现训练问题（如过拟合、欠拟合）

### 9. 测试函数详细实现

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)           # 测试样本总数
    num_batches = len(dataloader)            # 批次数量
    model.eval()                             # 设置为评估模式
    test_loss, correct = 0, 0

    with torch.no_grad():                    # 禁用梯度计算
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model.forward(x)          # 前向传播
            test_loss += loss_fn(pred, y).item()  # 累积损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # 计算指标
    test_loss /= num_batches                 # 平均损失
    correct /= size                          # 准确率百分比
    test_loss_history.append(test_loss)

    print(f"\nTest Error:\n")
    print(f"Accuracy: {(100*correct)}%")
    print(f"Avg loss: {test_loss}")
    show_samples(test_data, model, n_samples=10)
    return test_loss, correct
```

**评估过程详细分析**：

1. **model.eval()**：设置评估模式
   - 禁用Dropout
   - 使用批归一化的总体统计

2. **torch.no_grad()**：上下文管理器
   - 禁用梯度计算
   - 减少内存使用
   - 加速推理过程

3. **准确率计算**：
   ```python
   correct += (pred.argmax(1) == y).type(torch.float).sum().item()
   ```
   - `pred.argmax(1)` → 获取预测类别（最高logit值对应的索引）
   - `== y` → 与真实标签逐元素比较，返回布尔张量
   - `.type(torch.float)` → 将布尔值转换为浮点数（True→1.0, False→0.0）
   - `.sum()` → 统计正确预测的数量
   - `.item()` → 将PyTorch张量转换为Python数值

4. **指标计算**：
   - 所有批次的平均损失
   - 正确预测的百分比

### 10. 训练配置和执行

```python
loss_fn = nn.CrossEntropyLoss()                    # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0045)  # 优化器

if __name__ == "__main__":
    print(len(training_data))    # 60,000 训练样本
    print(len(test_data))        # 10,000 测试样本

    epochs = 25
    for i in range(epochs):
        print(f"\nEpoch {i+1}")
        train(train_dataloader, model, loss_fn, optimizer)

    test(test_dataloader, model, loss_fn)
```

**配置详细说明**：

1. **CrossEntropyLoss**：
   - 结合LogSoftmax和NLLLoss
   - 适用于多分类任务
   - 为学习提供强梯度信号

2. **Adam优化器**：
   - 学习率：0.0045（针对此问题调优）
   - 每个参数的自适应学习率
   - 内置动量和偏置修正

3. **训练循环**：
   - 25个轮次（完整遍历训练数据）
   - 每轮处理所有60,000个训练样本
   - 每轮打印进度信息
   - **每轮结束后自动绘制训练曲线**，实时监控训练效果

4. **实时可视化优势**：
   - 立即发现训练异常（损失不下降、震荡等）
   - 观察收敛速度，判断是否需要调整超参数
   - 及时停止训练，避免过拟合

## 总结与扩展

### 项目总结

本实现通过一个实际的MNIST分类任务展示了深度学习的基本概念。虽然多层感知机结构相对简单，但它有效地捕获了数字识别所需的模式。

### 关键学习要点

1. **架构重要性**：深度网络能够学习复杂的层次化特征
2. **正确训练**：仔细选择损失函数、优化器和超参数
3. **全面评估**：确保模型泛化能力的综合测试
4. **可视化理解**：通过视觉检查理解模型行为

### 深度学习核心概念回顾

#### 1. 神经网络基础
- **神经元**：基本计算单元，执行加权求和和激活
- **层**：神经元的集合，执行特定的变换
- **深度**：多层结构允许学习复杂的非线性映射

#### 2. 训练过程
- **前向传播**：数据从输入到输出的流动
- **损失计算**：量化预测与真实值的差异
- **反向传播**：计算梯度并更新参数
- **迭代优化**：重复上述过程直到收敛

#### 3. 关键技术
- **批处理**：提高训练效率和稳定性
- **优化器**：智能的参数更新策略
- **损失函数**：指导学习方向的目标函数

### 项目扩展建议

#### 从MLP到CNN：更适合图像处理的架构

虽然我们的多层感知机（MLP）在MNIST上表现不错，但对于图像处理任务，**卷积神经网络（CNN）**通常能获得更好的效果。

**MLP vs CNN的关键区别**：

1. **MLP的局限性**：
   - 将2D图像展平为1D向量，丢失了空间信息
   - 参数数量多，容易过拟合
   - 无法有效捕获图像的局部特征

2. **CNN的优势**：
   - 保持图像的2D结构，利用空间信息
   - 通过卷积核提取局部特征（边缘、纹理等）
   - 参数共享减少模型复杂度
   - 平移不变性，对图像位置变化更鲁棒

**CNN的基本组件**：
- **卷积层**：提取局部特征
- **池化层**：降低维度，增强鲁棒性
- **全连接层**：最终分类（类似我们的MLP部分）

在MNIST数据集上，CNN通常能达到99%以上的准确率，而我们的MLP大约在97%左右。这就是为什么在计算机视觉任务中，CNN是更主流的选择。

> **💡 学习建议**：掌握了MLP的基础原理后，CNN是自然的下一步。CNN本质上是在MLP前面加上了特征提取层，最后仍然使用全连接层进行分类。

### 常见问题和解决方案

#### 1. 训练问题
**问题**：损失不下降
**解决方案**：
- 检查学习率（可能太大或太小）
- 验证数据预处理
- 确认模型架构合理性

**问题**：过拟合
**解决方案**：
- 添加Dropout层
- 减少模型复杂度
- 增加训练数据
- 使用正则化技术

#### 2. 性能问题
**问题**：训练速度慢
**解决方案**：
- 使用GPU加速
- 增加批次大小
- 优化数据加载

**问题**：内存不足
**解决方案**：
- 减少批次大小
- 优化模型架构

## 结语

这个基础项目为您提供了深度学习的坚实起点。通过详细的代码解析和原理说明，相信您已经对神经网络的工作机制有了深入的理解。

**🚀 接下来的学习计划**：
- 掌握本项目的每个细节
- 尝试修改超参数，观察效果变化
- 为隐藏层添加激活函数（如ReLU）
- 准备学习CNN，这将是我们下一个教学项目

**📚 系列预告**：
后续我会陆续更新深度学习新手向代码教学系列，包括：
- CNN卷积神经网络详解
- RNN循环神经网络实战
- Transformer注意力机制
- 生成对抗网络GAN
- 更多实用的深度学习模型...

每个项目都会保持同样详细的代码教学风格，帮助初学者真正理解深度学习的精髓。

---

**作者**：[xiaoze]
**日期**：[2025-07-18]
**版本**：中文教学版 v1.0

**致谢**：感谢PyTorch团队提供优秀的深度学习框架，感谢MNIST数据集为深度学习教育做出的贡献。

