import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
import os
import time

# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# 定义ResNet18
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

# 训练函数
def train_and_evaluate(net, trainloader, testloader, epochs, lr, momentum):
    start_time = time.time()  # 记录开始时间
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100. * correct / total
        train_losses.append(train_loss / len(trainloader))
        train_accuracies.append(train_accuracy)

        # 测试
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_accuracy = 100. * correct / total
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.3f}, '
              f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

        scheduler.step()

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    return train_losses, train_accuracies, test_accuracies, total_time

if __name__ == '__main__':
    configs = [
        #{"epochs": 15, "lr": 0.01, "momentum": 0.9},        
        {"epochs": 15, "lr": 0.01, "momentum": 0.95},
        {"epochs": 25, "lr": 0.01, "momentum": 0.95},
        #{"epochs": 15, "lr": 0.001, "momentum": 0.9},
        #{"epochs": 15, "lr": 0.005, "momentum": 0.9},
        {"epochs": 25, "lr": 0.005, "momentum": 0.9},
        {"epochs": 25, "lr": 0.005, "momentum": 0.95},        
    ]

    results = {}

    for i, config in enumerate(configs):
        print(f"\nTraining with configuration {i+1}: {config}")
        
        net = ResNet18()
        train_losses, train_accuracies, test_accuracies, training_time = train_and_evaluate(
            net, trainloader, testloader, **config
        )

        results[f"config_{i+1}"] = {
            "config": config,
            "final_test_accuracy": test_accuracies[-1],
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "training_time": training_time
        }

        # 保存模型
        save_path = f'./modelset/Resnet/resnet18_config_{i+1}'
        os.makedirs(save_path, exist_ok=True)
        torch.save(net.state_dict(), f"{save_path}/final_model.pth")

    # 保存训练结果
    with open('./modelset/Resnet/Resnet_model_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nTraining completed. Results summary:")
    for config, result in results.items():
        print(f"\n{config}:")
        print(f"Configuration: {result['config']}")
        print(f"Final test accuracy: {result['final_test_accuracy']:.2f}%")
        print(f"Training time: {result['training_time']:.2f} seconds")
