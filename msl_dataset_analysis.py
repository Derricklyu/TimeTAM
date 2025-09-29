#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSL数据集深度分析脚本
该脚本用于生成MSL数据集的各种统计图表，包括数据分布、相关性分析、时间序列分析等。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 确保图表保存目录存在
output_dir = 'msl_analysis_charts'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class MSLDatasetAnalyzer:
    def __init__(self):
        # 加载数据集（使用模拟数据，实际应用中应替换为真实数据路径）
        self.train_data = None
        self.test_data = None
        self.test_labels = None
        self.load_dataset()
        self.prepare_data_for_visualization()
    
    def load_dataset(self):
        """加载MSL数据集"""
        # 由于无法直接访问真实数据集，这里使用模拟数据
        # 在实际使用时，请替换为实际的数据加载代码
        print("正在加载MSL数据集...")
        
        # 模拟数据参数
        train_samples = 58317
        test_samples = 73729
        features = 55
        
        # 创建符合统计特征的模拟数据
        np.random.seed(42)  # 设置随机种子以保证结果可复现
        
        # 模拟训练数据
        self.train_data = np.zeros((train_samples, features))
        # 第一个特征有实际变化
        self.train_data[:, 0] = np.random.normal(-0.49, 0.83, train_samples)
        self.train_data[:, 0] = np.clip(self.train_data[:, 0], -1.0, 2.2)
        
        # 其他特征大部分为0，只有少数样本有值
        for i in range(1, features):
            # 随机选择一小部分样本设置为非零值
            mask = np.random.random(train_samples) < 0.05
            self.train_data[mask, i] = 1.0
        
        # 模拟测试数据
        self.test_data = np.zeros((test_samples, features))
        self.test_data[:, 0] = np.random.normal(-0.49, 0.83, test_samples)
        self.test_data[:, 0] = np.clip(self.test_data[:, 0], -1.0, 2.2)
        
        for i in range(1, features):
            mask = np.random.random(test_samples) < 0.05
            self.test_data[mask, i] = 1.0
        
        # 模拟测试标签（10.53%异常样本）
        self.test_labels = np.zeros(test_samples)
        anomaly_count = int(test_samples * 0.1053)
        anomaly_indices = np.random.choice(test_samples, anomaly_count, replace=False)
        self.test_labels[anomaly_indices] = 1
        
        print(f"数据集加载完成：\n- 训练集: {train_samples}个样本, {features}个特征\n- 测试集: {test_samples}个样本, {features}个特征\n- 异常样本比例: {anomaly_count/test_samples*100:.2f}%")
    
    def prepare_data_for_visualization(self):
        """准备可视化所需的数据"""
        # 计算统计信息
        self.train_stats = {
            'mean': np.mean(self.train_data, axis=0),
            'std': np.std(self.train_data, axis=0),
            'min': np.min(self.train_data, axis=0),
            'max': np.max(self.train_data, axis=0),
            'median': np.median(self.train_data, axis=0)
        }
        
        # 使用numpy实现简单的数据标准化
        self.train_data_scaled = (self.train_data - self.train_stats['mean']) / (self.train_stats['std'] + 1e-8)  # 加1e-8避免除零
        self.test_data_scaled = (self.test_data - self.train_stats['mean']) / (self.train_stats['std'] + 1e-8)
        
        # 计算特征相关性
        self.correlation_matrix = np.corrcoef(self.train_data.T)
        
        # 选择前12个特征用于部分可视化（避免图表过于拥挤）
        self.top_features = 12
    
    def generate_sample_data_table(self):
        """生成样本数据表（返回HTML表格字符串）"""
        # 选择前5个训练样本和前5个测试样本
        train_samples = self.train_data[:5, :10]  # 只显示前10个特征
        test_samples = self.test_data[:5, :10]
        
        # 生成HTML表格
        html = "<table border='1' class='sample-table'>\n"
        html += "<thead>\n<tr><th>样本类型</th><th>样本ID</th>"
        for i in range(10):
            html += f"<th>特征{i+1}</th>"
        html += "</tr>\n</thead>\n"        
        html += "<tbody>\n"
        
        # 添加训练样本
        for i, sample in enumerate(train_samples):
            html += f"<tr><td>训练集</td><td>Train_{i}</td>"
            for val in sample:
                html += f"<td>{val:.4f}</td>"
            html += "</tr>\n"
        
        # 添加测试样本
        for i, sample in enumerate(test_samples):
            html += f"<tr><td>测试集</td><td>Test_{i}</td>"
            for val in sample:
                html += f"<td>{val:.4f}</td>"
            html += "</tr>\n"
        
        html += "</tbody>\n</table>"
        return html
    
    def plot_label_distribution(self):
        """绘制标签分布饼图和条形图"""
        # 计算正常和异常样本数量
        normal_count = np.sum(self.test_labels == 0)
        anomaly_count = np.sum(self.test_labels == 1)
        
        # 饼图
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.pie([normal_count, anomaly_count], labels=['正常样本', '异常样本'], autopct='%1.1f%%', 
                colors=['#66b3ff', '#ff6666'], startangle=90)
        plt.title('MSL测试集标签分布饼图')
        plt.axis('equal')
        
        # 条形图
        plt.subplot(122)
        bars = plt.bar(['正常样本', '异常样本'], [normal_count, anomaly_count], color=['#66b3ff', '#ff6666'])
        plt.title('MSL测试集标签分布条形图')
        plt.xlabel('样本类型')
        plt.ylabel('样本数量')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 100, 
                     f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成标签分布图")
    
    def plot_feature_distribution(self):
        """绘制特征分布直方图"""
        plt.figure(figsize=(15, 10))
        
        # 为前12个特征绘制直方图
        for i in range(self.top_features):
            plt.subplot(4, 3, i+1)
            plt.hist(self.train_data[:, i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'特征{i+1}分布')
            plt.xlabel('值')
            plt.ylabel('频数')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成特征分布图")
    
    def plot_time_series(self):
        """绘制时间序列图"""
        # 选择前500个时间步和前5个特征
        time_steps = 500
        features_to_plot = 5
        
        plt.figure(figsize=(15, 10))
        
        for i in range(features_to_plot):
            plt.subplot(features_to_plot, 1, i+1)
            plt.plot(self.test_data[:time_steps, i], 'b-', label=f'特征{i+1}')
            
            # 标记异常点
            anomaly_indices = np.where(self.test_labels[:time_steps] == 1)[0]
            if len(anomaly_indices) > 0:
                plt.scatter(anomaly_indices, self.test_data[anomaly_indices, i], 
                           color='red', s=20, label='异常点')
            
            plt.title(f'特征{i+1}时间序列')
            plt.ylabel('值')
            plt.legend()
        
        plt.xlabel('时间步')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成时间序列图")
    
    def plot_correlation_heatmap(self):
        """绘制特征相关性热力图"""
        plt.figure(figsize=(14, 12))
        
        # 只显示前20个特征的相关性矩阵
        corr_subset = self.correlation_matrix[:20, :20]
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', ['#0072B2', '#FFFFFF', '#D55E00'])
        
        # 使用matplotlib的imshow绘制热力图
        im = plt.imshow(corr_subset, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('相关系数')
        
        # 设置坐标轴标签
        plt.title('MSL数据集特征相关性热力图（前20个特征）')
        plt.xticks(np.arange(20), [f'特征{i+1}' for i in range(20)], rotation=45, ha='right')
        plt.yticks(np.arange(20), [f'特征{i+1}' for i in range(20)], rotation=0)
        
        # 添加网格线以区分单元格
        plt.grid(True, linestyle='-', color='white', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成相关性热力图")
    
    def plot_boxplots(self):
        """绘制特征箱线图"""
        plt.figure(figsize=(15, 10))
        
        # 选择前12个特征
        data_to_plot = [self.train_data[:, i] for i in range(self.top_features)]
        
        # 绘制箱线图
        box = plt.boxplot(data_to_plot, patch_artist=True, labels=[f'特征{i+1}' for i in range(self.top_features)])
        
        # 设置箱体颜色
        colors = plt.cm.viridis(np.linspace(0, 1, self.top_features))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('MSL数据集特征箱线图（前12个特征）')
        plt.xlabel('特征')
        plt.ylabel('值分布')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_feature_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成特征箱线图")
    
    def plot_radar_chart(self):
        """绘制特征统计信息雷达图"""
        # 选择前8个特征用于雷达图
        features_for_radar = 8
        
        # 准备数据
        stats_keys = ['min', 'max', 'mean', 'median', 'std']
        stats_labels = ['最小值', '最大值', '平均值', '中位数', '标准差']
        
        # 标准化每个特征的统计值以便比较
        normalized_stats = {}
        for key in stats_keys:
            min_val = np.min(self.train_stats[key][:features_for_radar])
            max_val = np.max(self.train_stats[key][:features_for_radar])
            if max_val - min_val > 0:
                normalized_stats[key] = (self.train_stats[key][:features_for_radar] - min_val) / (max_val - min_val)
            else:
                normalized_stats[key] = self.train_stats[key][:features_for_radar]
        
        # 绘制雷达图
        plt.figure(figsize=(12, 10))
        
        # 设置角度
        angles = np.linspace(0, 2*np.pi, features_for_radar, endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        ax = plt.subplot(111, polar=True)
        
        # 为每个统计指标绘制一条线
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (key, label) in enumerate(zip(stats_keys, stats_labels)):
            values = normalized_stats[key].tolist()
            values += values[:1]  # 闭合雷达图
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # 设置标签
        ax.set_thetagrids(np.degrees(angles[:-1]), [f'特征{i+1}' for i in range(features_for_radar)])
        ax.set_ylim(0, 1.1)
        ax.set_title('MSL数据集特征统计信息雷达图', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_feature_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成特征雷达图")
    
    def plot_anomaly_comparison(self):
        """比较正常样本和异常样本的特征分布"""
        # 提取正常和异常样本
        normal_samples = self.test_data[self.test_labels == 0]
        anomaly_samples = self.test_data[self.test_labels == 1]
        
        # 选择前4个特征进行比较
        features_to_compare = 4
        
        plt.figure(figsize=(15, 10))
        
        for i in range(features_to_compare):
            plt.subplot(2, 2, i+1)
            
            # 使用直方图代替KDE
            plt.hist(normal_samples[:, i], bins=30, alpha=0.5, label='正常样本', color='blue', density=True)
            plt.hist(anomaly_samples[:, i], bins=30, alpha=0.5, label='异常样本', color='red', density=True)
            
            plt.title(f'特征{i+1}：正常样本 vs 异常样本')
            plt.xlabel('值')
            plt.ylabel('密度')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_anomaly_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成正常/异常样本比较图")
    
    def plot_feature_activity_heatmap(self):
        """绘制特征活动热力图（显示哪些特征在何时有值）"""
        # 选择前500个时间步和前20个特征
        time_steps = 500
        features = 20
        
        # 创建活动矩阵（非零值为1，零值为0）
        activity_matrix = (self.test_data[:time_steps, :features] != 0).astype(int)
        
        plt.figure(figsize=(15, 10))
        
        # 使用matplotlib的imshow绘制热力图
        im = plt.imshow(activity_matrix.T, cmap='binary', aspect='auto', interpolation='nearest')
        
        plt.title('MSL数据集特征活动热力图（前20个特征，前500个时间步）')
        plt.xlabel('时间步')
        plt.ylabel('特征')
        
        # 设置刻度
        plt.xticks(np.arange(0, time_steps, 50))
        plt.yticks(np.arange(features))
        
        # 添加异常标记线
        anomaly_positions = np.where(self.test_labels[:time_steps] == 1)[0]
        for pos in anomaly_positions:
            plt.axvline(x=pos, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_feature_activity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成特征活动热力图")
    
    def plot_feature_statistics(self):
        """绘制特征统计信息对比图"""
        # 选择前8个特征
        features_to_plot = 8
        
        plt.figure(figsize=(15, 8))
        
        # 设置条形图位置
        bar_width = 0.2
        index = np.arange(features_to_plot)
        
        # 绘制均值、中位数、标准差对比图
        plt.bar(index - bar_width, self.train_stats['mean'][:features_to_plot], bar_width, label='平均值', color='blue')
        plt.bar(index, self.train_stats['median'][:features_to_plot], bar_width, label='中位数', color='green')
        plt.bar(index + bar_width, self.train_stats['std'][:features_to_plot], bar_width, label='标准差', color='orange')
        
        plt.title('MSL数据集特征统计信息对比（前8个特征）')
        plt.xlabel('特征')
        plt.ylabel('值')
        plt.xticks(index, [f'特征{i+1}' for i in range(features_to_plot)])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/msl_feature_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成特征统计信息对比图")
    
    def generate_all_charts(self):
        """生成所有图表"""
        self.plot_label_distribution()
        self.plot_feature_distribution()
        self.plot_time_series()
        self.plot_correlation_heatmap()
        self.plot_boxplots()
        self.plot_radar_chart()
        self.plot_anomaly_comparison()
        self.plot_feature_activity_heatmap()
        self.plot_feature_statistics()
        
        # 生成样本数据表格
        sample_table = self.generate_sample_data_table()
        
        # 保存样本数据表格到HTML文件
        with open(f'{output_dir}/sample_data_table.html', 'w', encoding='utf-8') as f:
            f.write(sample_table)
        
        print("所有图表生成完成！")
        print(f"图表保存在：{output_dir}")

if __name__ == "__main__":
    # 创建分析器实例
    analyzer = MSLDatasetAnalyzer()
    
    # 生成所有图表
    analyzer.generate_all_charts()