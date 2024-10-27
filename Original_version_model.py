import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import json
import os
show = ToPILImage()

# 设定对图片的归一化处理方式，并且下载数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

'''
# 观察一下数据集的内容
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 类别名称
print(len(trainset)) # 训练集大小
print(trainset[0][0].size()) # 第 1 条数据的图像大小
print(trainset[0][1]) # 第 1 条数据的标签
print(classes[trainset[0][1]]) # 第 1 条数据的文本标签

# 1 - 4行与上一块代码意义类似
(data, label) = trainset[12] # 选群训练集的一个样本展示内容，也可以改成其他数字看看
print(data.size()) 
print(label) # label是整数
print(classes[label])
show((data + 1) / 2).resize((100, 100)) # 还原被归一化的图片
'''


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        # 卷积层 '3'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5) 
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # 仿射层/全连接层，y = Wx + b
        self.fc1   = nn.Linear(16*5*5, 120) 
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        # 卷积 -> 激活 -> 池化 (relu激活函数不改变输入的形状)
        # [batch size, 3, 32, 32] -- conv1 --> [batch size, 6, 28, 28] -- maxpool --> [batch size, 6, 14, 14]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # [batch size, 6, 14, 14] -- conv2 --> [batch size, 16, 10, 10] --> maxpool --> [batch size, 16, 5, 5]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 把 16 * 5 * 5 的特征图展平，变为 [batch size, 16 * 5 * 5]，以送入全连接层
        x = x.view(x.size()[0], -1) 
        # [batch size, 16 * 5 * 5] -- fc1 --> [batch size, 120]
        x = F.relu(self.fc1(x))
        # [batch size, 120] -- fc2 --> [batch size, 84]
        x = F.relu(self.fc2(x))
        # [batch size, 84] -- fc3 --> [batch size, 10]
        x = self.fc3(x)        
        return x

net = Net()
#print(net)

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 使用SGD（随机梯度下降）优化
num_epochs = 5 # 训练 5 个 epoch

def train(trainloader, net, num_epochs, criterion, optimizer, save_path):
    losses = []
    for epoch in range(num_epochs):     
        running_loss = 0.0
        epoch_losses = []
        for i, data in enumerate(trainloader, 0):
    
            # 1. 取出数据
            inputs, labels = data
    
            # 梯度清零
            optimizer.zero_grad()
    
            # 2. 前向计算和反向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # 3. 反向传播，更新参数
            loss.backward()
            optimizer.step()

            # 记录损失
            running_loss += loss.item()
            if i % 1000 == 999:
                avg_loss = running_loss / 1000
                print('epoch %d: batch %5d loss: %.3f' \
                      % (epoch+1, i+1, avg_loss))
                epoch_losses.append({
                    'epoch': epoch + 1,
                    'batch': i + 1,
                    'loss': avg_loss
                })
                running_loss = 0.0
        
        losses.extend(epoch_losses)
        
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        torch.save(net.state_dict(), f"{save_path}/epoch_{epoch + 1}_model.pth")
    
    print('Finished Training')
    
    # 保存损失数据到文件
    with open(f"./modelset/original_version_model/training_losses.json", "w") as f:
        json.dump(losses, f)
    #with open(f"./modelset/original_version_model/training_losses.json", "w") as f:
    #    json.dump(losses, f, indent=4)
# 使用定义的网络进行训练
save_path = './modelset/original_version_model/weight_files'
train(trainloader, net, num_epochs, criterion, optimizer, save_path)

def predict(testloader, net):
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    
    with torch.no_grad(): # 正向传播时不计算梯度
        for data in testloader:
            # 1. 取出数据
            images, labels = data
            # 2. 正向传播，得到输出结果
            outputs = net(images)
            # 3. 从输出中得到模型预测
            _, predicted = torch.max(outputs, 1) # 1 表示沿着列的方向，返回两个值：最大值和对应的索引
            # 4. 计算性能指标
            total += labels.size(0) # size(0) 返回张量的第一个维度的大小，即当前批次中图片的数量
            correct += (predicted == labels).sum()
    
    print('测试集中的准确率为: %d %%' % (100 * correct / total))

predict(testloader, net)
