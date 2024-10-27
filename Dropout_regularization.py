import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import json
import os
import argparse
from datetime import datetime  # 修改这一行
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 如果使用GPU，打印GPU信息
if device.type == 'cuda':
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

# 设定对图片的归一化处理方式，并且下载数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

class Dropout_Net(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Dropout_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x

# 创建模型实例
def create_model(rank):
    model = Dropout_Net().to(rank)
    return model

def evaluate(net, dataloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train(rank, world_size, trainloader, testloader, net, num_epochs, criterion, optimizer, save_path):
    net = net.to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):     
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)
    
            optimizer.zero_grad()
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # 计算平均loss
        avg_loss = running_loss / len(trainloader)
        
        # 计算训练集和测试集准确率
        train_acc = evaluate(net, trainloader, rank)
        test_acc = evaluate(net, testloader, rank)
        
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            losses.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            })
            
            # 创建以 epoch 编号命名的权重文件
            os.makedirs(save_path, exist_ok=True)
            torch.save(net.module.state_dict(), f"{save_path}/epoch_{epoch + 1}_model.pth")
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    if rank == 0:
        print('Finished Training')
        with open(f"{save_path}/training_stats.json", "w") as f:
            json.dump(losses, f)
    
    return losses, train_accuracies, test_accuracies

def init_distributed():
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    return int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))

def get_device():
    if torch.cuda.is_available():
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    return torch.device('cpu')

def main():
    rank, world_size = init_distributed()
    device = get_device()
    print(f"Rank {rank}/{world_size} using device: {device}")
    print(f"Total number of GPUs being used: {world_size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                           download=True, transform=transform)

    # 定义不同的超参数配置
    configs = [
        #{"num_epochs": 50, "lr": 0.001, "momentum": 0.9, "batch_size": 32},
        {"num_epochs": 50, "lr": 0.005, "momentum": 0.9, "batch_size": 64},
        {"num_epochs": 50, "lr": 0.001, "momentum": 0.95, "batch_size": 32},
        #{"num_epochs": 50, "lr": 0.001, "momentum": 0.95, "batch_size": 64},
    ]

    results = {}

    for i, config in enumerate(configs):
        if rank == 0:
            print(f"\nTraining with configuration {i+1}: {config}")
        
        num_epochs = config["num_epochs"]
        lr = config["lr"]
        momentum = config["momentum"]
        batch_size = config["batch_size"]

        train_sampler = DistributedSampler(trainset) if world_size > 1 else None
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=(train_sampler is None),
                                                  sampler=train_sampler,
                                                  num_workers=2,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=True)

        net = create_model(rank)
        net = DDP(net, device_ids=[rank])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        save_path = f'./modelset/Dropout_regularization/config_{i+1}'
        if rank == 0:
            os.makedirs(save_path, exist_ok=True)

        train_losses, train_accuracies, test_accuracies = train(
            rank, world_size, trainloader, testloader, net, num_epochs, criterion, optimizer, save_path
        )

        results[f"config_{i+1}"] = {
            "config": config,
            "final_test_accuracy": test_accuracies[-1] if test_accuracies else None,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies
        }

    
    if rank == 0:
        # 创建目录
        os.makedirs('./modelset/Dropout_regularization_model', exist_ok=True)
        
        with open('./modelset/Dropout_regularization/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("\nTraining completed. Results summary:")
        for config, result in results.items():
            print(f"\n{config}:")
            print(f"Configuration: {result['config']}")
            print(f"Final test accuracy: {result['final_test_accuracy']:.2f}%" if result['final_test_accuracy'] is not None else "Final test accuracy: N/A")

    cleanup()

if __name__ == "__main__":
    main()
