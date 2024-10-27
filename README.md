# AI_homework
 本实验解决CIFAR-10 数据集上的图像分类问题。训练了两个神经网络Lenet和简化版本的Resnet，并对两个网络的超参数都做了专门调整，以提高分类效果。在Lenet上，模型最高准确率可以达到65.75%；在简化版本的Resnet上，模型最高准确率可以达到89.56%。
另外本实验实现了两种减少过拟合的正则化方法：L2正则化和Dropout 正则化，并且验证了正则化的效果。
四、实验（Experiments）
1.针对Task1,将project1.ipynb中提供的代码复制到Original_version_model.py中，作为实验的baseline。
实验设置（Experimental Setup）：CPU；数据集是CIFAR-10 数据集；模型是Lenet; 超参数设置(lr=0.001, momentum=0.9,num_epochs = 5)。
实验结果（Main Results）：训练后得到的相应的模型权重文件，训练过程loss的json文件，训练过程中loss的变化图像存储在modelset/original_version_model文件夹中。
训练五轮loss随训练整体呈下降趋势。测试集上的准确率60%。
<img width="358" alt="image" src="https://github.com/user-attachments/assets/a4a0c328-7ad4-4b8f-9432-afe9a3a8f1b8">


2.针对Task3,调整参数，考虑到原始版本的代码运行速度较慢，故想采用GPU并行训练加快训练速度，但原始版本的代码只支持CPU运算，无法直接用多张GPU并行训练，所以修改了代码，采用数据并行，并存储为Original_version_model_multiGPU.py。在4张A100上进行了测试，可以做到不损失准确率且加快训练速度。
实验设置（Experimental Setup）：4张A100；数据集是CIFAR-10 数据集；模型是Lenet; 超参数设置如下图。
实验结果（Main Results）：训练后得到的相应的模型权重文件，训练过程training_result的json文件，训练过程中loss,训练集上准确率和测试集上准确率的变化图像存储在modelset/original_version_model_multiGPU文件夹中。发现测试集上准确率最高可以达到65.75%（num_epochs=200,lr=0.001,momentum=0.95）；另外根据绘出的图观察到测试集上准确率随训练过程趋于一致，落在55%--70%的区间，继续调整超参数对模型预测准确率的提高并不明显。



3.针对Task2,分别试验L2正则化和Dropout 正则化方法，在Original_version_model_multiGPU.py基础上修改已加入正则化功能，将修改后的代码分别存储为L2_regularization.py和Dropout_ regularization.py。
实验设置（Experimental Setup）：4张A100；数据集是CIFAR-10 数据集；模型是正则化后的Lenet;
L2正则化：在配置中添加了 weight_decay 参数；在创建优化器时，将 weight_decay 参数传递给了 SGD 优化器：
这里的 weight_decay 参数实际上就是 L2 正则化系数 λ。当使用 weight_decay 时，优化器会在每次参数更新时自动将 L2 正则化项添加到损失函数中。具体来说，它会修改梯度更新规则如下：

这等效于在损失函数中添加了一个 L2 正则化项：

其中 ||w||^2 是模型参数的 L2 范数的平方。通过这种方式，L2 正则化鼓励模型学习更小的权重，从而减少过拟合的风险。weight_decay 的值（在这个例子中是 1e-4）控制了正则化的强度：值越大，正则化效果越强；值越小，正则化效果越弱。
Dropout正则化：在全连接层之间应用了 Dropout，这会在训练过程中随机"丢弃"一部分神经元，防止网络过度依赖某些特征，从而减少过拟合。默认的 Dropout 率设置为 0.5：

这意味着在每次前向传播时，有 50% 的神经元会被随机关闭。
实验结果（Main Results）：训练后得到的相应的模型权重文件，训练过程training_result的json文件存储在modelset/L2_regularization_model文件夹和modelset\Dropout_regularization中。根据验证两种正则化方法都有助于预防过拟合；但会略微降低准确率（下图为num_epochs: 50, lr: 0.001, momentum: 0.95）。 




4.针对Task4,借鉴Resnet,实现了一个18层的简化版本的Resnet,观察分类准确率和训练时间，并且也设置了不同的超参数config，调整num_epochs,lr,momentum这些参数，尽可能提高模型分类的性能。
实验设置（Experimental Setup）：1张A100；数据集是CIFAR-10 数据集；模型是18层的简化版本的Resnet; 超参数设置如下图。
实验结果（Main Results）：训练后得到的相应的模型权重文件，训练过程Resnet_model_result的json文件，训练过程中loss,训练集上准确率和测试集上准确率的变化图像存储在modelset/Resnet文件夹中。发现测试集上准确率最高可以达到89.56%（num_epochs=25,lr=0.005,momentum=0.95）。
模型选择的重要性大于对模型参数调整的重要性，实验中选择Resnet后效果明显好于Lenet。另外训练时间方面，由于使用GPU数量不同，直接比较单个模型最终训练时间意义不大，但1张A100上的Resnet与4张A100上的Lenet达到相同准确率时时间接近甚至更短，初步说明使用Resnet在模型准确率和训练时间方面均具有优越性。


