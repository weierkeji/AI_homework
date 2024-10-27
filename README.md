# CIFAR-10 图像分类实验

本项目包含了一系列使用不同模型和技术对CIFAR-10数据集进行图像分类的实验。

## 实验概览

1. 基线模型（LeNet on CPU）
2. 多GPU并行训练（LeNet on 4x A100）
3. 正则化技术（L2和Dropout）
4. 改进模型（简化版ResNet-18）

## 1. 基线模型

### 实验设置
- **硬件**：CPU
- **数据集**：CIFAR-10
- **模型**：LeNet
- **超参数**：
  - 学习率 (lr) = 0.001
  - 动量 (momentum) = 0.9
  - 训练轮数 (num_epochs) = 5

### 主要结果
- 模型权重、训练过程loss的JSON文件和loss变化图存储在 `modelset/original_version_model` 文件夹中
- 训练5轮后，loss整体呈下降趋势
- 测试集准确率：60%
  <img width="358" alt="image" src="https://github.com/user-attachments/assets/709bc60e-53c4-448a-a817-4e8bf50030b7">

## 2. 多GPU并行训练

### 实验设置
- **硬件**：4x NVIDIA A100 GPU
- **数据集**：CIFAR-10
- **模型**：LeNet（数据并行版本）
- **超参数**：见代码配置

### 主要结果
- 相关文件存储在 `modelset/original_version_model_multiGPU` 文件夹中
- 最高测试集准确率：65.75%（num_epochs=200, lr=0.001, momentum=0.95）
- 测试集准确率稳定在55%-70%区间
<img width="410" alt="image" src="https://github.com/user-attachments/assets/1e52ac64-0d39-493e-b17c-290c21c24079">
<img width="373" alt="image" src="https://github.com/user-attachments/assets/0b056041-e3b4-46c2-aa2c-207334ff121e">

## 3. 正则化技术

### 实验设置
- **硬件**：4x NVIDIA A100 GPU
- **数据集**：CIFAR-10
- **模型**：LeNet（加入正则化）
- **正则化方法**：
  1. L2正则化（weight_decay = 1e-4）
  2. Dropout正则化（dropout_rate = 0.5）

### 主要结果
- L2正则化结果存储在 `modelset/L2_regularization_model` 文件夹中
- Dropout正则化结果存储在 `modelset/Dropout_regularization` 文件夹中
- 两种方法都有助于预防过拟合，但略微降低了准确率
<img width="400" alt="image" src="https://github.com/user-attachments/assets/cd19c41b-3ce3-41d8-9ece-b03fefb0b94d">

## 4. 改进模型（ResNet-18）

### 实验设置
- **硬件**：1x NVIDIA A100 GPU
- **数据集**：CIFAR-10
- **模型**：简化版ResNet-18
- **超参数**：见代码配置

### 主要结果
- 相关文件存储在 `modelset/Resnet` 文件夹中
- 最高测试集准确率：89.56%（num_epochs=25, lr=0.005, momentum=0.95）
- ResNet模型在准确率和训练时间方面均优于LeNet
  <img width="391" alt="image" src="https://github.com/user-attachments/assets/3a10b6a1-f2a3-4d78-b840-4ef80b6a127b">
  <img width="399" alt="image" src="https://github.com/user-attachments/assets/05b77472-2f7e-4a2e-8da9-5af06a8fdaf9">

## 结论

1. 模型选择的重要性大于参数调整
2. ResNet在准确率和训练效率方面都优于LeNet
3. 正则化技术有助于防止过拟合，但可能略微降低准确率
4. 多GPU并行训练可以显著提高训练速度

## 文件结构

- `Original_version_model.py`: 基线模型（CPU版本）
- `Original_version_model_multiGPU.py`: 多GPU并行训练版本
- `L2_regularization.py`: 使用L2正则化的模型
- `Dropout_regularization.py`: 使用Dropout正则化的模型
- `Resnet_model.py`: 简化版ResNet-18模型

每个实验的详细结果和模型文件都存储在相应的 `modelset/` 子文件夹中。


