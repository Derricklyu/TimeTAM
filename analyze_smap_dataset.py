import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义数据集路径
dataset_dir = '/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/datasets/SMAP'

# 创建结果保存目录
result_dir = '/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/smap_analysis_results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

print("开始分析SMAP数据集...")

# 1. 读取并分析训练数据
train_data = np.load(os.path.join(dataset_dir, 'SMAP_train.npy'), allow_pickle=True)
print(f"\n训练数据信息:")
print(f"- 数据形状: {train_data.shape}")
print(f"- 数据类型: {train_data.dtype}")
print(f"- 数据维度: {train_data.ndim}")
print(f"- 样本数: {len(train_data)}")

# 2. 读取并分析测试数据
test_data = np.load(os.path.join(dataset_dir, 'SMAP_test.npy'), allow_pickle=True)
print(f"\n测试数据信息:")
print(f"- 数据形状: {test_data.shape}")
print(f"- 数据类型: {test_data.dtype}")
print(f"- 数据维度: {test_data.ndim}")
print(f"- 样本数: {len(test_data)}")

# 3. 读取并分析测试标签
test_labels = np.load(os.path.join(dataset_dir, 'SMAP_test_label.npy'), allow_pickle=True)
print(f"\n测试标签信息:")
print(f"- 标签形状: {test_labels.shape}")
print(f"- 标签类型: {test_labels.dtype}")
print(f"- 标签维度: {test_labels.ndim}")

# 4. 分析标签分布
label_counts = Counter(test_labels.flatten())
print(f"\n标签分布:")
for label, count in label_counts.items():
    print(f"- 标签 {label}: {count} 个样本 ({count/len(test_labels.flatten())*100:.2f}%)")

# 5. 分析数据统计特征
if train_data.ndim > 1:
    feature_dim = train_data.shape[1] if train_data.ndim == 2 else train_data.shape[-1]
    print(f"\n特征维度: {feature_dim}")
    
    # 计算部分样本的统计特征（避免计算量过大）
    sample_size = min(1000, len(train_data))
    train_sample = train_data[:sample_size]
    
    if train_sample.ndim == 3:
        # 如果是3D数据，展平为2D
        train_sample = train_sample.reshape(-1, train_sample.shape[-1])
    
    # 计算每个特征的统计量
    stats_df = pd.DataFrame()
    for i in range(feature_dim):
        feature_data = train_sample[:, i]
        stats_df[f'特征{i+1}'] = [np.min(feature_data), np.max(feature_data), np.mean(feature_data), 
                                 np.median(feature_data), np.std(feature_data)]
    
    stats_df.index = ['最小值', '最大值', '平均值', '中位数', '标准差']
    print(f"\n训练数据前{sample_size}个样本的统计特征:")
    print(stats_df)
    
    # 保存统计信息到CSV文件
    stats_df.to_csv(os.path.join(result_dir, 'smap_statistics.csv'), encoding='utf-8-sig')

# 6. 绘制标签分布饼图
plt.figure(figsize=(8, 6))
plt.pie(label_counts.values(), labels=[f'标签 {k}' for k in label_counts.keys()], autopct='%1.1f%%')
plt.title('SMAP测试集标签分布')
plt.savefig(os.path.join(result_dir, 'smap_label_distribution.png'))
plt.close()

# 7. 绘制数据样本的时间序列图（选择前3个特征的前200个时间步）
plt.figure(figsize=(12, 8))
if test_data.ndim == 3:
    # 对于3D数据，取第一个测试样本
    sample_data = test_data[0]
    sample_labels = test_labels[0] if test_labels.ndim == 3 else test_labels[:len(sample_data)]
    
    # 绘制前3个特征
    for i in range(min(3, sample_data.shape[1])):
        plt.subplot(3, 1, i+1)
        plt.plot(sample_data[:, i])
        # 标记异常点
        anomaly_indices = np.where(sample_labels > 0)[0]
        if len(anomaly_indices) > 0:
            plt.scatter(anomaly_indices, sample_data[anomaly_indices, i], color='red', s=10, label='异常点')
        plt.title(f'测试样本特征 {i+1} 的时间序列')
        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.grid(True)
        plt.legend()
elif test_data.ndim == 2:
    # 对于2D数据，假设是多个时间序列样本
    sample_data = test_data[:200]  # 取前200个样本
    sample_labels = test_labels[:200]
    
    # 绘制前3个特征（如果有的话）
    for i in range(min(3, sample_data.shape[1])):
        plt.subplot(3, 1, i+1)
        plt.plot(sample_data[:, i])
        # 标记异常点
        anomaly_indices = np.where(sample_labels > 0)[0]
        if len(anomaly_indices) > 0:
            plt.scatter(anomaly_indices, sample_data[anomaly_indices, i], color='red', s=10, label='异常点')
        plt.title(f'前200个样本特征 {i+1} 的时间序列')
        plt.xlabel('样本索引')
        plt.ylabel('数值')
        plt.grid(True)
        plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'smap_time_series_example.png'))
plt.close()

# 8. 绘制特征相关性热图（如果数据维度合适）
if train_data.ndim == 2 and train_data.shape[1] <= 20:
    # 对于2D数据，计算特征相关性
    sample_size = min(1000, len(train_data))
    correlation_matrix = np.corrcoef(train_data[:sample_size].T)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='相关系数')
    plt.title('SMAP训练数据特征相关性热图')
    plt.xticks(range(train_data.shape[1]), [f'特征{i+1}' for i in range(train_data.shape[1])], rotation=45)
    plt.yticks(range(train_data.shape[1]), [f'特征{i+1}' for i in range(train_data.shape[1])])
    plt.savefig(os.path.join(result_dir, 'smap_feature_correlation.png'))
    plt.close()

# 9. 输出文件以生成完整报告
report_path = os.path.join(result_dir, 'smap_data_analysis_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# SMAP数据集分析报告\n\n")
    
    f.write("## 1. 数据集基本信息\n")
    f.write(f"- **数据格式**: .npy (NumPy数组)\n")
    f.write(f"- **训练集大小**: {train_data.shape}\n")
    f.write(f"- **测试集大小**: {test_data.shape}\n")
    f.write(f"- **测试标签大小**: {test_labels.shape}\n")
    f.write(f"- **数据维度**: {train_data.ndim}维\n\n")
    
    f.write("## 2. 数据分布特征\n")
    f.write(f"### 2.1 标签分布\n")
    for label, count in label_counts.items():
        percentage = count/len(test_labels.flatten())*100
        f.write(f"- 标签 {label}: {count} 个样本 ({percentage:.2f}%)\n")
    f.write("\n")
    
    f.write("### 2.2 数据统计特征\n")
    f.write("训练数据部分样本的统计特征如下：\n")
    f.write("![数据统计特征](smap_statistics.csv)\n\n")
    
    f.write("## 3. 数据可视化\n")
    f.write("### 3.1 标签分布图\n")
    f.write("![标签分布](smap_label_distribution.png)\n\n")
    
    f.write("### 3.2 时间序列示例\n")
    f.write("下图展示了测试数据中部分样本的时间序列特征，红色点标记为异常点：\n")
    f.write("![时间序列示例](smap_time_series_example.png)\n\n")
    
    if train_data.ndim == 2 and train_data.shape[1] <= 20:
        f.write("### 3.3 特征相关性热图\n")
        f.write("![特征相关性热图](smap_feature_correlation.png)\n\n")
    
    f.write("## 4. 数据特点总结\n")
    if len(label_counts) > 1:
        anomaly_count = label_counts.get(1, 0) + label_counts.get(-1, 0)
        normal_count = label_counts.get(0, 0)
        if anomaly_count < normal_count:
            f.write("- 数据集为典型的异常检测数据集，异常样本数量较少（不平衡数据）\n")
        else:
            f.write("- 数据集标签分布较为均衡\n")
    
    if 'stats_df' in locals():
        # 分析数据是否标准化
        max_std = stats_df.loc['标准差'].max()
        min_std = stats_df.loc['标准差'].min()
        if max_std > 10 and min_std > 1:
            f.write("- 数据可能未经过标准化处理，各特征的标准差差异较大\n")
        else:
            f.write("- 数据可能已经过标准化处理\n")
    
    f.write("- 数据为多维时间序列数据，适合用于异常检测任务\n")
    f.write("- 测试集包含标签信息，可用于评估模型性能\n\n")
    
    f.write("## 5. 后续实验建议\n")
    f.write("1. 数据预处理：根据统计特征考虑是否需要标准化或归一化\n")
    f.write("2. 模型选择：适合使用时间序列异常检测算法\n")
    f.write("3. 评估指标：建议使用精确率、召回率、F1分数等指标评估模型性能\n")

print(f"\nSMAP数据集分析完成！")
print(f"分析报告和图表已保存至：{result_dir}")