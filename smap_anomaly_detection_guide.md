# SMAP数据集异常检测实验指南

本指南提供了在Mac电脑上使用GPU加速运行SMAP数据集异常检测实验的详细步骤，以及如何生成专业的实验报告。

## 实验概述

本实验基于VQ-VAE (Vector Quantized Variational Autoencoder) 模型对SMAP数据集进行异常检测。实验流程包括以下几个主要阶段：
1. 数据预处理 - 将原始数据转换为模型可处理的格式
2. RevIN转换 - 应用Reversible Instance Normalization进行数据标准化
3. VQ-VAE模型训练 - 训练用于异常检测的模型
4. 异常检测与报告生成 - 使用训练好的模型检测异常并生成可视化报告

## 环境准备

在运行实验之前，请确保您的Mac环境中已安装以下软件包：
- Python 3.x
- PyTorch (支持MPS后端)
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

您可以使用以下命令安装所需的依赖：
```bash
pip install torch numpy matplotlib seaborn scikit-learn
```

## 实验运行步骤

### 步骤1：修改文件权限

首先，确保shell脚本具有执行权限：

```bash
chmod +x /Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/anomaly_detection/scripts/smap_mac.sh
```

### 步骤2：检查数据集路径

确保SMAP数据集文件位于以下路径：
- `/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/datasets/SMAP/SMAP_train.npy`
- `/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/datasets/SMAP/SMAP_test.npy`
- `/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/datasets/SMAP/SMAP_test_label.npy`

如果您的数据集路径不同，请修改`smap_mac.sh`脚本中的`data_root_path`变量。

### 步骤3：运行实验脚本

执行以下命令运行完整的实验流程：

```bash
cd /Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main
./anomaly_detection/scripts/smap_mac.sh
```

## 脚本功能说明

### smap_mac.sh

这个shell脚本整合了实验的四个主要阶段：

1. **数据预处理**：调用`save_chunked_data.py`将原始数据转换为模型训练所需的格式
2. **RevIN转换**：调用`revin_data.py`对数据进行标准化处理
3. **模型训练**：调用`train_vqvae.py`训练VQ-VAE模型，使用Mac的GPU加速
4. **异常检测与报告**：调用`run_smap_anomaly_detection.py`进行异常检测并生成专业的HTML报告

### run_smap_anomaly_detection.py

这个Python脚本实现了以下功能：

1. **异常检测**：使用训练好的VQ-VAE模型对测试数据进行异常检测
2. **结果分析**：计算准确率、精确率、召回率和F1分数等评估指标
3. **可视化生成**：创建多种可视化图表，包括：
   - 误差分布图
   - 异常检测结果示例
   - 码本使用情况统计
   - 码本嵌入的t-SNE可视化
   - 时间序列重构示例
   - 混淆矩阵
4. **HTML报告生成**：整合所有分析结果和可视化图表，生成专业的HTML报告

## 输出结果说明

实验完成后，您将在以下路径找到所有输出结果：

```
/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/smap_anomaly_detection_report/
```

该目录包含：

1. **smap_anomaly_detection_report.html** - 完整的实验报告，包含所有分析结果和可视化图表
2. **experiment_results.json** - 实验结果的JSON格式数据，方便后续分析
3. **可视化图表文件** - 各种PNG格式的图表文件，包括：
   - error_distribution.png - 测试集误差分布图
   - anomaly_detection_example.png - 异常检测结果示例图
   - codebook_usage.png - 码本使用情况统计图
   - code_embedding_tsne.png - 码本嵌入的t-SNE可视化图
   - reconstruction_example.png - 时间序列重构示例图
   - confusion_matrix.png - 混淆矩阵图

## 结果解释指南

### 实验指标

HTML报告中包含的主要评估指标：

- **准确率(Accuracy)**：模型正确分类的样本比例
- **精确率(Precision)**：在预测为异常的样本中，实际为异常的比例
- **召回率(Recall)**：在实际为异常的样本中，被模型正确预测的比例
- **F1分数**：精确率和召回率的调和平均数，综合反映模型性能

### 可视化图表解释

1. **误差分布与检测阈值**：展示测试数据的重构误差分布和用于异常检测的阈值

2. **异常检测结果示例**：展示前500个时间点的重构误差和异常检测结果，直观展示模型性能

3. **码本使用情况**：展示VQ-VAE模型中各个码本向量的使用频率，反映模型的表示能力

4. **码本嵌入可视化**：使用t-SNE算法将高维码本嵌入降维到二维空间，展示码本的聚类特性

5. **时间序列重构示例**：对比原始数据和模型重构数据，展示模型对时间序列的建模能力

6. **混淆矩阵**：直观展示模型在正常样本和异常样本上的分类表现

## 注意事项

1. **GPU加速**：脚本会自动检测并使用Mac的GPU(MPS)进行计算加速

2. **运行时间**：完整实验可能需要较长时间，具体取决于您的硬件性能

3. **内存使用**：处理大型数据集时可能需要较多内存，请确保您的系统有足够的可用内存

4. **结果可重复性**：实验使用固定种子(47)以确保结果的可重复性

5. **报告查看**：生成的HTML报告可以用任何现代浏览器打开查看

## 问题排查

如果实验过程中遇到问题，请检查以下几点：

1. 确保所有依赖包已正确安装
2. 检查数据集文件路径是否正确
3. 确保您的Mac支持MPS加速（通常需要macOS 12.3或更高版本）
4. 检查系统内存是否充足

如果问题仍然存在，请查看终端输出的错误信息，这将有助于定位具体问题。

---

本指南旨在帮助您在Mac环境下顺利运行SMAP数据集的异常检测实验，并理解实验结果。如有任何问题或需要进一步的帮助，请参考代码注释或联系技术支持。