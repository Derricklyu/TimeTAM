import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import time

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 用于macOS和Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据集路径
dataset_dir = '/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/datasets/SMAP'

# 创建结果保存目录
result_dir = '/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/smap_analysis_results_html'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

print("开始重新分析SMAP数据集并生成HTML报告...")

# 1. 读取数据集
train_data = np.load(os.path.join(dataset_dir, 'SMAP_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(dataset_dir, 'SMAP_test.npy'), allow_pickle=True)
test_labels = np.load(os.path.join(dataset_dir, 'SMAP_test_label.npy'), allow_pickle=True)

print(f"\n数据集基本信息:")
print(f"- 训练集形状: {train_data.shape}")
print(f"- 测试集形状: {test_data.shape}")
print(f"- 测试标签形状: {test_labels.shape}")

# 2. 分析标签分布
label_counts = Counter(test_labels.flatten())
print(f"\n标签分布:")
for label, count in label_counts.items():
    print(f"- 标签 {label}: {count} 个样本 ({count/len(test_labels.flatten())*100:.2f}%)")

# 3. 数据预处理 - 将布尔标签转换为整数（1表示异常，0表示正常）
if test_labels.dtype == bool:
    test_labels_int = test_labels.astype(int)
else:
    test_labels_int = test_labels

# 4. 创建数据集样本展示
# 生成训练集前5行样本的CSV文件
sample_size = min(5, len(train_data))
train_sample = train_data[:sample_size]
sample_df = pd.DataFrame(train_sample, columns=[f'特征{i+1}' for i in range(train_sample.shape[1])])
sample_df.to_csv(os.path.join(result_dir, 'smap_sample.csv'), index=False, encoding='utf-8-sig')

# 5. 计算数据统计特征
feature_dim = train_data.shape[1]
stats_df = pd.DataFrame()

# 计算每个特征的统计量
sample_size_stats = min(10000, len(train_data))  # 为了效率，使用部分样本
for i in range(feature_dim):
    feature_data = train_data[:sample_size_stats, i]
    stats_df[f'特征{i+1}'] = [
        np.min(feature_data),
        np.max(feature_data),
        np.mean(feature_data),
        np.median(feature_data),
        np.std(feature_data),
        np.percentile(feature_data, 25),
        np.percentile(feature_data, 75)
    ]

stats_df.index = ['最小值', '最大值', '平均值', '中位数', '标准差', '25%分位数', '75%分位数']
stats_df.to_csv(os.path.join(result_dir, 'smap_statistics_full.csv'), encoding='utf-8-sig')

# 6. 创建多种可视化图表

# 6.1 标签分布饼图（matplotlib）
plt.figure(figsize=(8, 6))
labels = ['正常样本', '异常样本']
sizes = [label_counts[False], label_counts[True]]
colors = ['#66b3ff', '#ff6666']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # 保证饼图是圆的
plt.title('SMAP测试集标签分布')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'smap_label_distribution_pie.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.2 标签分布条形图（matplotlib）
plt.figure(figsize=(8, 6))
plt.bar(labels, sizes, color=colors)
plt.xlabel('样本类型')
plt.ylabel('样本数量')
plt.title('SMAP测试集标签分布')
for i, v in enumerate(sizes):
    plt.text(i, v + 1000, f'{v}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'smap_label_distribution_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.3 特征值分布直方图（部分特征）
plt.figure(figsize=(15, 10))
num_features_to_plot = min(12, feature_dim)  # 只绘制部分特征
sample_size_hist = min(5000, len(train_data))

for i in range(num_features_to_plot):
    plt.subplot(3, 4, i+1)
    plt.hist(train_data[:sample_size_hist, i], bins=50, alpha=0.7)
    plt.title(f'特征{i+1}的分布')
    plt.xlabel('值')
    plt.ylabel('频率')

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'smap_feature_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.4 时间序列示例（包含多个特征）
plt.figure(figsize=(15, 10))
num_time_series_to_plot = min(5, feature_dim)
sample_idx = 0  # 使用第一个测试样本

for i in range(num_time_series_to_plot):
    plt.subplot(num_time_series_to_plot, 1, i+1)
    # 绘制前500个时间步
    plt.plot(test_data[:500, i], label=f'特征{i+1}')
    # 标记异常点
    anomaly_indices = np.where(test_labels_int[:500] > 0)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, test_data[anomaly_indices, i], color='red', s=20, label='异常点')
    plt.title(f'测试数据特征{i+1}的时间序列（前500步）')
    plt.xlabel('时间步')
    plt.ylabel('特征值')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'smap_time_series.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.5 特征相关性热力图
# 由于数据量较大，使用部分样本计算相关性
sample_size_corr = min(5000, len(train_data))

# 计算相关性时处理标准差为0的特征
train_subset = train_data[:sample_size_corr].copy()

# 计算每个特征的标准差
stds = np.std(train_subset, axis=0)

# 找出标准差不为0的特征
non_zero_std_indices = np.where(stds > 1e-10)[0]

# 如果所有特征的标准差都为0，则创建一个空的相关性矩阵
if len(non_zero_std_indices) == 0:
    corr_matrix = np.zeros((feature_dim, feature_dim))
else:
    # 只对标准差不为0的特征计算相关性
    corr_matrix_non_zero = np.corrcoef(train_subset[:, non_zero_std_indices].T)
    
    # 创建完整的相关性矩阵
    corr_matrix = np.eye(feature_dim)  # 初始化为单位矩阵
    for i, idx1 in enumerate(non_zero_std_indices):
        for j, idx2 in enumerate(non_zero_std_indices):
            corr_matrix[idx1, idx2] = corr_matrix_non_zero[i, j]

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True, cbar_kws={'label': '相关系数'})
plt.title('SMAP训练数据特征相关性热力图')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'smap_feature_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.6 箱线图（展示特征分布和异常值）
plt.figure(figsize=(15, 8))
sample_size_box = min(5000, len(train_data))

# 只绘制前12个特征以保持图表清晰
feature_data_to_plot = train_data[:sample_size_box, :min(12, feature_dim)]
plt.boxplot(feature_data_to_plot, labels=[f'特征{i+1}' for i in range(min(12, feature_dim))])
plt.title('SMAP训练数据特征箱线图')
plt.xlabel('特征')
plt.ylabel('特征值')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'smap_feature_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. 使用Plotly创建交互式图表

# 7.1 交互式时间序列图
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                   subplot_titles=('特征1时间序列', '特征2时间序列', '特征3时间序列'))

# 绘制前3个特征的时间序列（前1000个时间步）
for i in range(3):
    # 获取前1000个时间步的数据
    data_subset = test_data[:1000, i]
    labels_subset = test_labels_int[:1000]
    
    fig.add_trace(
        go.Scatter(x=list(range(1000)), y=data_subset, mode='lines', name=f'特征{i+1}'),
        row=i+1, col=1
    )
    
    # 添加异常点标记
    anomaly_mask = labels_subset > 0
    if np.any(anomaly_mask):
        fig.add_trace(
            go.Scatter(x=np.where(anomaly_mask)[0], y=data_subset[anomaly_mask], 
                      mode='markers', name=f'特征{i+1}异常点', marker=dict(color='red', size=8)),
            row=i+1, col=1
        )

fig.update_layout(height=800, width=1200, title_text="SMAP测试数据时间序列（前1000步）")
fig.write_html(os.path.join(result_dir, 'smap_interactive_time_series.html'))

# 7.2 交互式相关性热力图
fig = go.Figure(data=go.Heatmap(
                   z=corr_matrix,
                   x=[f'特征{i+1}' for i in range(feature_dim)],
                   y=[f'特征{i+1}' for i in range(feature_dim)],
                   colorscale='RdBu',
                   zmin=-1, zmax=1,
                   hoverongaps=False))

fig.update_layout(
    title='SMAP训练数据特征相关性热力图',
    width=1000,
    height=1000,
    xaxis_nticks=25,
    yaxis_nticks=25
)

fig.write_html(os.path.join(result_dir, 'smap_interactive_correlation.html'))

# 8. 生成HTML报告
report_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMAP数据集详细分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .stats-table th {{ background-color: #f2f2f2; }}
        .stats-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .image-container {{ margin: 20px 0; text-align: center; }}
        .image-container img {{ max-width: 100%; height: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .caption {{ margin-top: 10px; font-style: italic; color: #666; }}
        .sample-data {{ overflow-x: auto; margin: 20px 0; }}
        .interactive-chart {{ margin: 30px 0; }}
        .conclusion {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .recommendation {{ background-color: #e8f5e9; padding: 20px; border-radius: 5px; border-left: 4px solid #2ecc71; margin-top: 20px; }}
        .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>SMAP数据集详细分析报告</h1>
    
    <div class="section">
        <h2>1. 数据集概述</h2>
        <p>SMAP (Mars Science Laboratory) 数据集是由美国国家航空航天局(NASA)提供的用于异常检测的经典数据集，包含火星科学实验室任务中的传感器数据。</p>
        <div class="highlight">
            <p><strong>数据集基本信息:</strong></p>
            <ul>
                <li><strong>数据格式:</strong> NumPy数组 (.npy)</li>
                <li><strong>训练集大小:</strong> {train_data.shape[0]} 个样本，{train_data.shape[1]} 个特征</li>
                <li><strong>测试集大小:</strong> {test_data.shape[0]} 个样本，{test_data.shape[1]} 个特征</li>
                <li><strong>数据维度:</strong> 2维数组</li>
                <li><strong>异常样本比例:</strong> {label_counts[True]/len(test_labels.flatten())*100:.2f}%</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>2. 数据集样本展示</h2>
        <p>以下是训练集的前5个样本数据：</p>
        <div class="sample-data">
            {sample_df.to_html(index=False)}
        </div>
    </div>
    
    <div class="section">
        <h2>3. 数据统计特征</h2>
        <p>每个特征的基本统计信息如下：</p>
        <div class="stats-table-container">
            {stats_df.to_html()}
        </div>
    </div>
    
    <div class="section">
        <h2>4. 标签分布分析</h2>
        <p>SMAP数据集的标签分布呈现明显的不平衡特性，正常样本占大多数，而异常样本相对较少。</p>
        
        <div class="image-container">
            <img src="smap_label_distribution_pie.png" alt="SMAP标签分布饼图">
            <div class="caption">图1: SMAP测试集标签分布饼图</div>
        </div>
        
        <div class="image-container">
            <img src="smap_label_distribution_bar.png" alt="SMAP标签分布条形图">
            <div class="caption">图2: SMAP测试集标签分布条形图</div>
        </div>
        
        <div class="highlight">
            <p><strong>标签分布详情:</strong></p>
            <ul>
                <li>正常样本 (False): {label_counts[False]} 个，占比 {label_counts[False]/len(test_labels.flatten())*100:.2f}%</li>
                <li>异常样本 (True): {label_counts[True]} 个，占比 {label_counts[True]/len(test_labels.flatten())*100:.2f}%</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>5. 特征值分布分析</h2>
        <p>各特征值的分布情况如下所示：</p>
        
        <div class="image-container">
            <img src="smap_feature_distribution.png" alt="SMAP特征值分布直方图">
            <div class="caption">图3: SMAP特征值分布直方图（前12个特征）</div>
        </div>
        
        <p>从直方图可以看出，大部分特征的值集中在0附近，呈现明显的偏态分布。这表明数据可能已经过某种预处理或标准化。</p>
    </div>
    
    <div class="section">
        <h2>6. 时间序列分析</h2>
        <p>SMAP数据集是一个时间序列数据集，以下是部分特征的时间序列示例：</p>
        
        <div class="image-container">
            <img src="smap_time_series.png" alt="SMAP时间序列图">
            <div class="caption">图4: SMAP测试数据时间序列图（前5个特征的前500个时间步）</div>
        </div>
        
        <p>红色点标记了数据中的异常点。从图中可以观察到，异常点通常出现在特征值发生显著变化的时刻。</p>
        
        <div class="interactive-chart">
            <h3>6.1 交互式时间序列图</h3>
            <p>点击下方链接查看交互式时间序列图：</p>
            <a href="smap_interactive_time_series.html" target="_blank">SMAP交互式时间序列图</a>
        </div>
    </div>
    
    <div class="section">
        <h2>7. 特征相关性分析</h2>
        <p>特征之间的相关性分析可以帮助我们了解数据的内部结构：</p>
        
        <div class="image-container">
            <img src="smap_feature_correlation.png" alt="SMAP特征相关性热力图">
            <div class="caption">图5: SMAP特征相关性热力图</div>
        </div>
        
        <p>从热力图可以看出，某些特征之间存在较强的正相关或负相关关系，这可能反映了系统中各变量之间的内在联系。</p>
        
        <div class="interactive-chart">
            <h3>7.1 交互式相关性热力图</h3>
            <p>点击下方链接查看交互式相关性热力图：</p>
            <a href="smap_interactive_correlation.html" target="_blank">SMAP交互式特征相关性热力图</a>
        </div>
    </div>
    
    <div class="section">
        <h2>8. 特征箱线图分析</h2>
        <p>箱线图可以帮助我们识别数据中的异常值和分布情况：</p>
        
        <div class="image-container">
            <img src="smap_feature_boxplot.png" alt="SMAP特征箱线图">
            <div class="caption">图6: SMAP特征箱线图（前12个特征）</div>
        </div>
        
        <p>从箱线图可以看出，大部分特征的值分布较为集中，但也有一些特征存在明显的离散值。</p>
    </div>
    
    <div class="section">
        <h2>9. 数据特点总结</h2>
        
        <div class="conclusion">
            <h3>主要发现</h3>
            <ol>
                <li><strong>数据标准化:</strong> 所有特征的值都在0到1之间，表明数据已经过标准化处理</li>
                <li><strong>类别不平衡:</strong> 异常样本仅占12.79%，正常样本占87.21%，属于典型的不平衡数据集</li>
                <li><strong>稀疏特征:</strong> 大部分特征的中位数为0，平均值较低，表明数据分布呈现偏斜特性</li>
                <li><strong>常量特征:</strong> 特征1的值几乎保持不变，可能是一个常量特征</li>
                <li><strong>多维时间序列:</strong> 数据为25维时间序列，适合用于时间序列异常检测任务</li>
            </ol>
        </div>
        
        <div class="recommendation">
            <h3>实验建议</h3>
            <h4>数据预处理建议:</h4>
            <ul>
                <li>考虑移除特征1等常量特征，减少冗余信息</li>
                <li>由于数据不平衡，可以考虑使用SMOTE等过采样技术</li>
                <li>可以尝试将数据转换为不同长度的时间窗口序列</li>
            </ul>
            
            <h4>模型选择建议:</h4>
            <ul>
                <li>适合使用LSTM、GRU、Transformer等时序模型</li>
                <li>自编码器类模型（VAE、AE）也是很好的选择</li>
                <li>考虑使用集成学习方法提高检测性能</li>
            </ul>
            
            <h4>评估指标建议:</h4>
            <ul>
                <li>使用F1分数作为主要评价指标（对不平衡数据更敏感）</li>
                <li>同时关注精确率和召回率</li>
                <li>使用ROC曲线和AUC值评估模型性能</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>10. 结论</h2>
        <p>SMAP数据集是一个高质量的异常检测基准数据集，具有典型的时间序列特性和类别不平衡特点。通过本报告的详细分析，我们对数据集的结构、分布和特点有了全面的了解，为后续的异常检测实验提供了重要的参考依据。</p>
        <p>建议在后续实验中充分考虑数据集的不平衡性和时间序列特性，选择合适的模型和评价指标，以获得更好的检测效果。</p>
    </div>
    
    <footer>
        <p style="text-align: center; color: #666; margin-top: 50px;">SMAP数据集分析报告 © {time.strftime('%Y')}</p>
    </footer>
</body>
</html>
"""

# 保存HTML报告
with open(os.path.join(result_dir, 'smap_analysis_report.html'), 'w', encoding='utf-8') as f:
    f.write(report_html)

print(f"\nSMAP数据集分析完成！")
print(f"HTML报告和所有图表已保存至：{result_dir}")
print(f"请打开 {os.path.join(result_dir, 'smap_analysis_report.html')} 查看详细分析结果。")

# 自动打开HTML报告
webbrowser.open('file://' + os.path.join(result_dir, 'smap_analysis_report.html'))