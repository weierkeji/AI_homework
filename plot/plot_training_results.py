import json
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open('./modelset/Original_version_model_multiGPU/training_results.json', 'r') as f:
    results = json.load(f)

# 为每个配置创建图表
for config, data in results.items():
    epochs = [entry['epoch'] for entry in data['train_losses']]
    loss = [entry['loss'] for entry in data['train_losses']]
    train_acc = [entry['train_acc'] for entry in data['train_losses']]
    test_acc = [entry['test_acc'] for entry in data['train_losses']]

    # 创建准确率图表
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='blue')
    plt.plot(epochs, test_acc, label='Test Accuracy', color='green')
    plt.title(f'Training and Test Accuracy for {config}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 添加配置信息到图表
    config_info = data['config']
    info_text = f"LR: {config_info['lr']}, Momentum: {config_info['momentum']}, Batch Size: {config_info['batch_size']}"
    plt.text(0.05, 0.05, info_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='bottom')

    # 保存准确率图表
    plt.savefig(f'./modelset/Original_version_model_multiGPU/{config}_accuracy_plot.png')
    plt.close()

    # 创建损失图表
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, label='Loss', color='red')
    plt.title(f'Training Loss for {config}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 添加配置信息到图表
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')

    # 保存损失图表
    plt.savefig(f'./modelset/Original_version_model_multiGPU/{config}_loss_plot.png')
    plt.close()

print("Plots have been saved in the modelset/Original_version_model_multiGPU directory.")
