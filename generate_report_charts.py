#!/usr/bin/env python3
"""
使用matplotlib生成异常检测报告所需的所有图表，并保存为高分辨率图片。
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置图表保存目录
output_dir = 'report_charts'
os.makedirs(output_dir, exist_ok=True)

# 设置图表分辨率和尺寸
dpi = 300
figsize = (10, 6)

print(f"图表将保存在：{output_dir}")
print(f"图表分辨率：{dpi} dpi")
print(f"图表尺寸：{figsize[0]}x{figsize[1]} 英寸")

# 1. 特征分布直方图
def generate_feature_distribution():
    plt.figure(figsize=figsize)
    
    # 模拟数据 - 与原报告中ECharts的数据一致
    bins = ['-1.5~-1.0', '-1.0~-0.5', '-0.5~0.0', '0.0~0.5', '0.5~1.0', '1.0~1.5', '1.5~2.0', '2.0~4.5']
    frequencies = [150, 230, 290, 180, 90, 40, 15, 5]
    
    plt.bar(bins, frequencies, color='#3498db', alpha=0.8)
    plt.title('特征值分布直方图', fontsize=16)
    plt.xlabel('特征值范围', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'feature_distribution.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 2. 时间序列示例
def generate_time_series():
    plt.figure(figsize=figsize)
    
    # 模拟数据 - 基于原报告中的均值和标准差
    time_points = np.arange(200)
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    time_series_data = -0.0889 + 0.9050 * np.sin(time_points * 0.1) + 0.1 * np.random.randn(200)
    
    plt.plot(time_points[:50], time_series_data[:50], color='#e74c3c', linewidth=2)
    plt.title('时间序列示例', fontsize=16)
    plt.xlabel('时间点', fontsize=12)
    plt.ylabel('特征值', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'time_series.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 3. 标签分布饼图
def generate_label_distribution():
    plt.figure(figsize=(8, 8))
    
    # 数据 - 正常85%，异常15%
    labels = ['正常', '异常']
    sizes = [85, 15]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.1)  # 突出显示异常部分
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    plt.axis('equal')  # 保证饼图是正圆形
    plt.title('标签分布', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'label_distribution.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 4. VQVAE模型架构图
def generate_vqvae_architecture():
    plt.figure(figsize=(12, 6))
    
    # 简单的架构示意图
    # 定义组件位置
    components = {
        '输入': (0.1, 0.5),
        '编码器': (0.3, 0.5),
        '量化层': (0.5, 0.5),
        '解码器': (0.7, 0.5),
        '输出': (0.9, 0.5),
        '码本': (0.5, 0.7)
    }
    
    # 绘制组件
    for name, (x, y) in components.items():
        plt.scatter(x, y, s=500, color='#3498db', alpha=0.8)
        plt.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 绘制连接线
    connections = [
        ('输入', '编码器'),
        ('编码器', '量化层'),
        ('量化层', '解码器'),
        ('解码器', '输出'),
        ('量化层', '码本'),
        ('码本', '量化层')
    ]
    
    for source, target in connections:
        x1, y1 = components[source]
        x2, y2 = components[target]
        
        # 普通实线连接
        if source == '量化层' and target == '码本' or source == '码本' and target == '量化层':
            # 虚线连接
            plt.plot([x1, x2], [y1, y2], 'k--', linewidth=1.5)
        else:
            # 实线连接
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        
        # 添加箭头
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        plt.arrow(x1 + 0.1*dx, y1 + 0.1*dy, 0.8*dx, 0.8*dy, 
                  head_width=0.02, head_length=0.03, fc='k', ec='k', linewidth=2)
    
    plt.xticks([])
    plt.yticks([])
    plt.title('VQVAE模型架构', fontsize=16)
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    save_path = os.path.join(output_dir, 'vqvae_architecture.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 5. 码本可视化
def generate_codebook_visualization():
    plt.figure(figsize=figsize)
    
    # 生成模拟的码本向量并使用t-SNE降维
    np.random.seed(42)
    codebook = np.random.randn(100, 64)  # 100个64维的向量
    
    # 使用t-SNE降维到2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    codebook_2d = tsne.fit_transform(codebook)
    
    # 归一化坐标以便更好地显示
    codebook_2d = (codebook_2d - codebook_2d.min(axis=0)) / (codebook_2d.max(axis=0) - codebook_2d.min(axis=0))
    codebook_2d = codebook_2d * 10 - 5  # 映射到[-5,5]范围
    
    # 绘制散点图，使用不同颜色区分不同的向量
    colors = plt.cm.hsv(np.linspace(0, 1, 100))
    scatter = plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], c=colors, s=50, alpha=0.8)
    
    plt.title('码本向量可视化（t-SNE降维后）', fontsize=16)
    plt.xlabel('维度1', fontsize=12)
    plt.ylabel('维度2', fontsize=12)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'codebook_visualization.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 6. 训练损失曲线
def generate_training_loss():
    plt.figure(figsize=figsize)
    
    # 模拟训练损失数据
    iterations = np.arange(0, 3001, 100)
    loss_data = [1.2, 0.8, 0.6, 0.5, 0.45, 0.42, 0.39, 0.37, 0.35, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12]
    
    plt.plot(iterations, loss_data, color='#9b59b6', linewidth=2)
    plt.title('训练损失曲线', fontsize=16)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 7. 重构误差分布
def generate_reconstruction_error():
    plt.figure(figsize=figsize)
    
    # 模拟数据
    error_bins = ['0.0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0', '1.0~1.2', '1.2~1.4', '1.4~1.6', '1.6~1.8', '1.8~2.0']
    normal_freq = [350, 280, 180, 100, 60, 20, 10, 0, 0, 0]
    anomaly_freq = [10, 20, 30, 40, 60, 80, 100, 90, 70, 50]
    
    width = 0.35
    x = np.arange(len(error_bins))
    
    plt.bar(x - width/2, normal_freq, width, label='正常样本', color='#2ecc71', alpha=0.8)
    plt.bar(x + width/2, anomaly_freq, width, label='异常样本', color='#e74c3c', alpha=0.8)
    
    plt.title('重构误差分布', fontsize=16)
    plt.xlabel('重构误差', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.xticks(x, error_bins, rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'reconstruction_error.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 8. ROC曲线图
def generate_roc_curve():
    plt.figure(figsize=figsize)
    
    # 模拟ROC曲线数据
    fpr = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    tpr = [0, 0.3, 0.45, 0.55, 0.65, 0.7, 0.75, 0.8, 0.83, 0.86, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0]
    
    # 计算AUC值
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC曲线', fontsize=16)
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 9. Token使用频率分布图
def generate_token_usage():
    plt.figure(figsize=figsize)
    
    # 模拟Token使用频率数据
    token_indices = np.arange(0, 1000, 50)
    token_freq = np.exp(-(token_indices - 500)**2 / (2 * 200**2)) * 1000 + np.random.randn(len(token_indices)) * 100
    token_freq = np.maximum(token_freq, 0)  # 确保频率不为负
    
    # 使用渐变色
    colors = plt.cm.hsv(np.linspace(0, 1, len(token_indices)))
    
    plt.bar(token_indices, token_freq, color=colors, alpha=0.8)
    plt.title('Token使用频率分布', fontsize=16)
    plt.xlabel('Token索引', fontsize=12)
    plt.ylabel('使用频率', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'token_usage.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 10. 模型性能指标柱状图
def generate_performance_metrics():
    plt.figure(figsize=figsize)
    
    # 实际实验结果数据
    metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC值']
    values = [0.949, 0.8457, 0.9127, 0.8779, 0.96]
    
    # 绘制柱状图
    bars = plt.bar(metrics, values, color=['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.8)
    
    # 在柱子上方添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metric == 'F1分数':
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('异常检测性能指标', fontsize=16)
    plt.ylabel('值', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'performance_metrics.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已生成：{save_path}")

# 执行所有图表生成函数
def main():
    print("开始生成报告图表...")
    
    # 检查是否安装了必要的库
    try:
        import sklearn
    except ImportError:
        print("错误：未安装sklearn库，请先运行 'pip install scikit-learn'")
        return
    
    # 生成所有图表
    generate_feature_distribution()
    generate_time_series()
    generate_label_distribution()
    generate_vqvae_architecture()
    generate_codebook_visualization()
    generate_training_loss()
    generate_reconstruction_error()
    generate_roc_curve()
    generate_token_usage()
    
    print("所有图表生成完成！")
    print(f"图表保存在：{os.path.abspath(output_dir)}")
    print("请使用这些图片替换HTML报告中的ECharts图表。")

if __name__ == "__main__":
    main()