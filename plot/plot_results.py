import json
import matplotlib.pyplot as plt

# 读取JSON文件
with open('./modelset/Original_version_model_multiGPU/training_results.json', 'r') as f:
    results = json.load(f)

# 设置颜色
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']

# 创建三个图表
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# 为每个配置创建带参数的标签
labels = []
for i, (config, data) in enumerate(results.items()):
    params = data['config']
    label = f"Config {i+1}: epochs={params['num_epochs']}, lr={params['lr']}, momentum={params['momentum']}, batch_size={params['batch_size']}"
    labels.append(label)

# 绘制训练损失
for i, (config, data) in enumerate(results.items()):
    losses = [entry['loss'] for entry in data['train_losses']]
    ax1.plot(range(1, len(losses) + 1), losses, color=colors[i], label=labels[i])
ax1.set_title('Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='x-small')

# 绘制训练准确率
for i, (config, data) in enumerate(results.items()):
    ax2.plot(range(1, len(data['train_accuracies']) + 1), data['train_accuracies'], color=colors[i], label=f"Config {i+1}")
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.legend(loc='lower right')

# 绘制测试准确率
for i, (config, data) in enumerate(results.items()):
    ax3.plot(range(1, len(data['test_accuracies']) + 1), data['test_accuracies'], color=colors[i], label=f"Config {i+1}")
ax3.set_title('Test Accuracy')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Accuracy (%)')
ax3.legend(loc='lower right')

# 调整子图之间的间距和整体布局
plt.tight_layout()

# 保存图表
plt.savefig('./modelset/Original_version_model_multiGPU/training_results_plots.png', bbox_inches='tight', dpi=300)
print("Plots have been saved to ./modelset/Original_version_model_multiGPU/training_results_plots.png")

# 显示图表（可选）
# plt.show()
