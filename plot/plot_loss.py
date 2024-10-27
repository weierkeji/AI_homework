import json
import matplotlib.pyplot as plt

def plot_loss(loss_file):
    # 从文件中读取损失数据
    with open(loss_file, 'r') as f:
        losses = json.load(f)
    
    # 提取数据
    epochs = [item['epoch'] for item in losses]
    batches = [item['batch'] for item in losses]
    loss_values = [item['loss'] for item in losses]
    
    # 计算总batch数
    total_batches = [(epoch - 1) * max(batches) + batch for epoch, batch in zip(epochs, batches)]
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(total_batches, loss_values, marker='.')
    plt.title('Training Loss over Batches')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 添加epoch分隔线
    for epoch in range(1, max(epochs) + 1):
        plt.axvline(x=(epoch-1)*max(batches), color='r', linestyle='--', alpha=0.5)
        plt.text((epoch-1)*max(batches), plt.ylim()[1], f'Epoch {epoch}', 
                 horizontalalignment='right', verticalalignment='top')
    
    # 保存图像
    plt.savefig('./modelset/original_version_model/loss_curve.png')
    #plt.show()

if __name__ == "__main__":
    loss_file = './modelset/original_version_model/training_losses.json'
    plot_loss(loss_file)
