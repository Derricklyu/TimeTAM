import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import json
import re
from datetime import datetime
import sys
from pathlib import Path
import pickle
import math
from tqdm import tqdm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class DatasetAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.output_dir = os.path.join(root_dir, 'dataset_analysis_reports')
        os.makedirs(self.output_dir, exist_ok=True)
        self.dataset_stats = {}
        self.dataset_paths = {
            'MSL': os.path.join(root_dir, 'datasets', 'MSL'),
            'PSM': os.path.join(root_dir, 'datasets', 'PSM'),
            'PSM2': os.path.join(root_dir, 'datasets', 'PSM 2'),
            'SMAP': os.path.join(root_dir, 'datasets', 'SMAP'),
            'SMD': os.path.join(root_dir, 'datasets', 'SMD'),
            'SWaT': os.path.join(root_dir, 'datasets', 'SWaT')
        }
        
        # 创建一个HTML模板
        self.html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据集分析报告</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, SimHei, sans-serif; margin: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .section { margin-bottom: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }
        .chart-container { position: relative; height:300px; margin: 20px 0; }
        .stats-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        .stats-table th { background-color: #4CAF50; color: white; }
        .dataset-card { margin-bottom: 30px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
        .dataset-header { background-color: #343a40; color: white; padding: 15px; }
        .dataset-content { padding: 20px; }
        .sample-data { overflow-x: auto; margin: 20px 0; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .image-item { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
        .image-item img { max-width: 100%; height: auto; max-height: 300px; }
        .analysis-section { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-5">数据集分析报告</h1>
        <div class="section">
            <h2>概述</h2>
            <p>本报告对以下数据集进行了全面分析：{dataset_names}</p>
            <div class="stats-summary">
                {summary_stats}
            </div>
        </div>
        
        {dataset_sections}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.min.js"></script>
</body>
</html>"""
        
        self.dataset_section_template = """
        <div class="dataset-card">
            <div class="dataset-header">
                <h2>{dataset_name}</h2>
            </div>
            <div class="dataset-content">
                <h3>基本信息</h3>
                <p>{basic_info}</p>
                
                <h3>数据结构</h3>
                <div class="stats-table-container">
                    <table class="stats-table">
                        <tr>
                            <th>数据类型</th>
                            <th>文件</th>
                            <th>样本数</th>
                            <th>特征数</th>
                            <th>其他维度</th>
                        </tr>
                        {data_structure_rows}
                    </table>
                </div>
                
                <h3>统计指标</h3>
                <div class="stats-table-container">
                    <table class="stats-table">
                        <tr>
                            <th>特征</th>
                            <th>均值</th>
                            <th>标准差</th>
                            <th>最小值</th>
                            <th>25%分位数</th>
                            <th>中位数</th>
                            <th>75%分位数</th>
                            <th>最大值</th>
                        </tr>
                        {stats_rows}
                    </table>
                </div>
                
                <h3>数据样本</h3>
                <div class="sample-data">
                    <table class="stats-table">
                        {sample_rows}
                    </table>
                </div>
                
                <h3>数据情况分析</h3>
                <div class="analysis-section">
                    {data_analysis}
                </div>
                
                <h3>可视化分析</h3>
                <div class="image-grid">
                    {visualization_images}
                </div>
                
                <h3>数据处理建议</h3>
                <div class="analysis-section">
                    {data_recommendations}
                </div>
            </div>
        </div>"""
    
    def generate_data_analysis(self, data_dict, dataset_name):
        """生成数据情况分析"""
        analysis = []
        
        try:
            # 检查是否有训练数据
            if 'train' not in data_dict:
                return "<p>缺少训练数据，无法进行完整分析。</p>"
            
            train_data = data_dict['train']['data']
            
            # 检查数据类型并转换为pandas DataFrame以便分析
            if isinstance(train_data, np.ndarray):
                # 处理多维数组
                if len(train_data.shape) > 2:
                    # 对于多维时间序列数据，我们只分析第一个时间步
                    df = pd.DataFrame(train_data[:, :, 0])
                else:
                    df = pd.DataFrame(train_data)
                
                # 使用特征名称
                feature_names = data_dict.get('feature_names', [f'feature_{i}' for i in range(df.shape[1])])
                df.columns = feature_names[:df.shape[1]]
            elif isinstance(train_data, pd.DataFrame):
                df = train_data.copy()
            else:
                return "<p>不支持的数据类型，无法进行完整分析。</p>"
            
            # 1. 检查缺失值
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                analysis.append(f"<p>✓ 数据中存在缺失值，总数：{missing_values}</p>")
                # 找出缺失值较多的特征
                missing_by_col = df.isnull().sum()
                high_missing = missing_by_col[missing_by_col > 0].head(5)
                if not high_missing.empty:
                    analysis.append("<p>缺失值较多的前5个特征：</p>")
                    analysis.append("<ul>")
                    for col, count in high_missing.items():
                        percentage = (count / len(df)) * 100
                        analysis.append(f"<li>{col}: {count} ({percentage:.2f}%)</li>")
                    analysis.append("</ul>")
            else:
                analysis.append("<p>✓ 数据中没有缺失值</p>")
            
            # 2. 检测异常值（使用IQR方法）
            outliers_info = []
            for column in df.select_dtypes(include=[np.number]).columns:
                if df[column].nunique() > 1:  # 确保列不是常量
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        outlier_percentage = (outlier_count / len(df)) * 100
                        outliers_info.append((column, outlier_count, outlier_percentage))
            
            if outliers_info:
                # 按异常值比例排序并取前5个
                outliers_info.sort(key=lambda x: x[2], reverse=True)
                analysis.append(f"<p>✓ 检测到异常值，主要集中在以下特征：</p>")
                analysis.append("<ul>")
                for col, count, percentage in outliers_info[:5]:
                    analysis.append(f"<li>{col}: {count} ({percentage:.2f}%)</li>")
                analysis.append("</ul>")
            else:
                analysis.append("<p>✓ 使用IQR方法未检测到明显的异常值</p>")
            
            # 3. 数据分布分析
            analysis.append("<p>✓ 数据分布情况：</p>")
            analysis.append("<ul>")
            
            # 偏度分析
            skewness = df.select_dtypes(include=[np.number]).skew()
            high_skew = skewness[abs(skewness) > 2]
            if not high_skew.empty:
                analysis.append(f"<li>发现 {len(high_skew)} 个高度偏斜的特征（偏度绝对值>2）</li>")
            else:
                analysis.append("<li>特征分布较为对称，无明显偏斜</li>")
            
            # 方差分析
            variance = df.select_dtypes(include=[np.number]).var()
            zero_variance = variance[variance == 0]
            if not zero_variance.empty:
                analysis.append(f"<li>发现 {len(zero_variance)} 个零方差特征（常量特征）</li>")
            
            # 数据范围分析
            min_values = df.select_dtypes(include=[np.number]).min()
            max_values = df.select_dtypes(include=[np.number]).max()
            ranges = max_values - min_values
            large_range = ranges[ranges > 1000]
            if not large_range.empty:
                analysis.append(f"<li>发现 {len(large_range)} 个数据范围较大的特征，可能需要标准化处理</li>")
            
            analysis.append("</ul>")
            
            # 4. 标签分布分析（如果有标签）
            if 'test_labels' in data_dict:
                test_labels = data_dict['test_labels']['data']
                if isinstance(test_labels, np.ndarray):
                    labels_flat = test_labels.flatten()
                    normal_count = int(np.sum(labels_flat == 0))
                    anomaly_count = int(np.sum(labels_flat == 1))
                    total = len(labels_flat)
                    
                    analysis.append(f"<p>✓ 标签分布：正常样本 {normal_count} ({normal_count/total*100:.2f}%)，异常样本 {anomaly_count} ({anomaly_count/total*100:.2f}%)</p>")
                    
                    # 检查类别不平衡
                    if anomaly_count / total < 0.05:
                        analysis.append("<p><strong>注意：数据存在严重的类别不平衡问题，异常样本比例不足5%</strong></p>")
                
            return ''.join(analysis)
        except Exception as e:
            print(f"生成{dataset_name}数据情况分析出错: {e}")
            return f"<p>数据情况分析生成过程中出错：{str(e)}</p>"
    
    def generate_data_recommendations(self, data_dict, dataset_name):
        """生成数据处理建议"""
        recommendations = []
        
        try:
            # 检查是否有训练数据
            if 'train' not in data_dict:
                return "<p>缺少训练数据，无法提供具体建议。</p>"
            
            train_data = data_dict['train']['data']
            
            # 检查数据类型并转换为pandas DataFrame以便分析
            if isinstance(train_data, np.ndarray):
                # 处理多维数组
                if len(train_data.shape) > 2:
                    df = pd.DataFrame(train_data[:, :, 0])
                else:
                    df = pd.DataFrame(train_data)
                
                # 使用特征名称
                feature_names = data_dict.get('feature_names', [f'feature_{i}' for i in range(df.shape[1])])
                df.columns = feature_names[:df.shape[1]]
            elif isinstance(train_data, pd.DataFrame):
                df = train_data.copy()
            else:
                return "<p>不支持的数据类型，无法提供具体建议。</p>"
            
            recommendations.append("<ol>")
            
            # 1. 缺失值处理建议
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                recommendations.append("<li><strong>缺失值处理</strong>：")
                missing_percentage = (missing_values / (df.shape[0] * df.shape[1])) * 100
                if missing_percentage < 5:
                    recommendations.append("考虑使用均值、中位数或众数填充缺失值；对于时间序列数据，可以考虑前向填充或后向填充。")
                else:
                    recommendations.append("缺失值比例较高，建议先分析缺失原因，然后考虑使用插值法、模型预测法填充，或使用能够处理缺失值的算法。")
                recommendations.append("</li>")
            
            # 2. 异常值处理建议
            # 计算异常值
            has_outliers = False
            for column in df.select_dtypes(include=[np.number]).columns:
                if df[column].nunique() > 1:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
                    if len(outliers) > 0:
                        has_outliers = True
                        break
            
            if has_outliers:
                recommendations.append("<li><strong>异常值处理</strong>：")
                recommendations.append("建议先分析异常值的来源，区分真实异常和数据错误。对于数据错误，可以考虑：")
                recommendations.append("<ul>")
                recommendations.append("<li>使用IQR或Z-score方法识别并删除极端异常值</li>")
                recommendations.append("<li>将异常值截断到合理范围（ Winsorization 方法）</li>")
                recommendations.append("<li>对于时间序列数据，可以使用移动平均等方法平滑处理</li>")
                recommendations.append("</ul>")
                recommendations.append("</li>")
            
            # 3. 数据标准化/归一化建议
            # 检查数据范围
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                min_values = numeric_df.min()
                max_values = numeric_df.max()
                ranges = max_values - min_values
                large_range = ranges[ranges > 1000]
                
                # 检查特征间的尺度差异
                std_devs = numeric_df.std()
                if len(std_devs) > 1 and (std_devs.max() / std_devs.min()) > 10:
                    recommendations.append("<li><strong>数据标准化</strong>：")
                    recommendations.append("特征间存在较大的尺度差异，建议进行标准化处理：")
                    recommendations.append("<ul>")
                    recommendations.append("<li>如果数据近似正态分布，使用StandardScaler进行Z-score标准化</li>")
                    recommendations.append("<li>如果数据分布偏斜或有异常值，使用RobustScaler</li>")
                    recommendations.append("<li>如果需要将数据映射到[0,1]区间，使用MinMaxScaler</li>")
                    recommendations.append("</ul>")
                    recommendations.append("</li>")
                elif not large_range.empty:
                    recommendations.append("<li><strong>数据缩放</strong>：")
                    recommendations.append("部分特征数据范围较大，建议进行适当缩放以提高模型训练效果。")
                    recommendations.append("</li>")
            
            # 4. 特征工程建议
            recommendations.append("<li><strong>特征工程</strong>：")
            recommendations.append("<ul>")
            recommendations.append("<li>对于时间序列数据，可以考虑提取统计特征（如均值、方差、峰值等）或频域特征</li>")
            recommendations.append("<li>可以尝试特征选择方法（如PCA、LASSO回归等）减少特征维度</li>")
            recommendations.append("<li>考虑创建新的衍生特征，如滑动窗口统计量</li>")
            recommendations.append("</ul>")
            recommendations.append("</li>")
            
            # 5. 模型选择建议
            recommendations.append("<li><strong>模型选择</strong>：")
            recommendations.append("基于数据集特点，建议尝试以下模型：")
            recommendations.append("<ul>")
            
            # 检查是否有标签和异常样本比例
            if 'test_labels' in data_dict:
                test_labels = data_dict['test_labels']['data']
                if isinstance(test_labels, np.ndarray):
                    labels_flat = test_labels.flatten()
                    anomaly_count = int(np.sum(labels_flat == 1))
                    total = len(labels_flat)
                    
                    if anomaly_count / total < 0.05:
                        # 严重的类别不平衡
                        recommendations.append("<li>异常检测算法：Isolation Forest、One-Class SVM</li>")
                        recommendations.append("<li>考虑使用SMOTE等方法进行数据增强</li>")
                        recommendations.append("<li>调整模型参数以处理类别不平衡，如增加少数类的权重</li>")
                    else:
                        recommendations.append("<li>分类算法：Random Forest、XGBoost、LightGBM</li>")
                        recommendations.append("<li>深度学习方法：LSTM、GRU等循环神经网络</li>")
            else:
                recommendations.append("<li>无监督学习算法：聚类分析、自编码器</li>")
                recommendations.append("<li>异常检测算法：LOF、EllipticEnvelope</li>")
            
            recommendations.append("</ul>")
            recommendations.append("</li>")
            
            # 6. 交叉验证建议
            recommendations.append("<li><strong>模型评估</strong>：")
            recommendations.append("使用合适的评估指标和验证策略：")
            recommendations.append("<ul>")
            recommendations.append("<li>对于时间序列数据，使用滚动窗口验证而非随机分割</li>")
            recommendations.append("<li>评估指标：准确率、精确率、召回率、F1分数、ROC曲线</li>")
            recommendations.append("<li>对于不平衡数据，重点关注召回率和精确率的平衡</li>")
            recommendations.append("</ul>")
            recommendations.append("</li>")
            
            recommendations.append("</ol>")
            
            return ''.join(recommendations)
        except Exception as e:
            print(f"生成{dataset_name}数据处理建议出错: {e}")
            return f"<p>数据处理建议生成过程中出错：{str(e)}</p>"
    
    def load_dataset(self, dataset_name, dataset_path):
        """根据数据集名称加载对应的数据"""
        data_dict = {}
        
        if dataset_name in ['MSL', 'SMAP']:
            # 加载.npy格式的数据
            try:
                train_data = np.load(os.path.join(dataset_path, f'{dataset_name}_train.npy'))
                test_data = np.load(os.path.join(dataset_path, f'{dataset_name}_test.npy'))
                test_labels = np.load(os.path.join(dataset_path, f'{dataset_name}_test_label.npy'))
                
                data_dict['train'] = {'data': train_data, 'type': 'npy'}
                data_dict['test'] = {'data': test_data, 'type': 'npy'}
                data_dict['test_labels'] = {'data': test_labels, 'type': 'npy'}
                
                # 生成特征名称
                num_features = train_data.shape[1] if len(train_data.shape) > 1 else 1
                data_dict['feature_names'] = [f'feature_{i}' for i in range(num_features)]
            except Exception as e:
                print(f"加载{dataset_name}数据集出错: {e}")
        
        elif dataset_name == 'PSM' or dataset_name == 'PSM2':
            # 加载.csv格式的数据
            try:
                train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
                test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
                test_labels = pd.read_csv(os.path.join(dataset_path, 'test_label.csv'))
                
                data_dict['train'] = {'data': train_data, 'type': 'csv'}
                data_dict['test'] = {'data': test_data, 'type': 'csv'}
                data_dict['test_labels'] = {'data': test_labels, 'type': 'csv'}
                
                # 使用实际的列名作为特征名称
                data_dict['feature_names'] = list(train_data.columns)
            except Exception as e:
                print(f"加载{dataset_name}数据集出错: {e}")
        
        elif dataset_name == 'SMD':
            # 加载.npy和.pkl格式的数据
            try:
                train_data = np.load(os.path.join(dataset_path, 'SMD_train.npy'))
                test_data = np.load(os.path.join(dataset_path, 'SMD_test.npy'))
                test_labels = np.load(os.path.join(dataset_path, 'SMD_test_label.npy'))
                
                # 尝试加载pkl文件获取更多信息
                try:
                    with open(os.path.join(dataset_path, 'SMD_train.pkl'), 'rb') as f:
                        train_pkl = pickle.load(f)
                    with open(os.path.join(dataset_path, 'SMD_test.pkl'), 'rb') as f:
                        test_pkl = pickle.load(f)
                    with open(os.path.join(dataset_path, 'SMD_test_label.pkl'), 'rb') as f:
                        test_labels_pkl = pickle.load(f)
                except:
                    pass
                
                data_dict['train'] = {'data': train_data, 'type': 'npy'}
                data_dict['test'] = {'data': test_data, 'type': 'npy'}
                data_dict['test_labels'] = {'data': test_labels, 'type': 'npy'}
                
                # 生成特征名称
                num_features = train_data.shape[1] if len(train_data.shape) > 1 else 1
                data_dict['feature_names'] = [f'feature_{i}' for i in range(num_features)]
            except Exception as e:
                print(f"加载{dataset_name}数据集出错: {e}")
        
        elif dataset_name == 'SWaT':
            # SWaT数据集可能有多种格式
            try:
                # 检查是否有train.csv文件
                if os.path.exists(os.path.join(dataset_path, 'swat_train.csv')):
                    train_data = pd.read_csv(os.path.join(dataset_path, 'swat_train.csv'), low_memory=False)
                    data_dict['train'] = {'data': train_data, 'type': 'csv'}
                    data_dict['feature_names'] = list(train_data.columns)
                
                # 检查是否有test文件
                if os.path.exists(os.path.join(dataset_path, 'swat2.csv')):
                    test_data = pd.read_csv(os.path.join(dataset_path, 'swat2.csv'), low_memory=False)
                    data_dict['test'] = {'data': test_data, 'type': 'csv'}
                    if 'feature_names' not in data_dict:
                        data_dict['feature_names'] = list(test_data.columns)
                
                # 检查原始数据文件
                if os.path.exists(os.path.join(dataset_path, 'swat_raw.csv')):
                    raw_data = pd.read_csv(os.path.join(dataset_path, 'swat_raw.csv'), low_memory=False)
                    data_dict['raw'] = {'data': raw_data, 'type': 'csv'}
                    if 'feature_names' not in data_dict:
                        data_dict['feature_names'] = list(raw_data.columns)
            except Exception as e:
                print(f"加载{dataset_name}数据集出错: {e}")
        
        return data_dict
    
    def analyze_dataset(self, dataset_name, dataset_path):
        """分析单个数据集"""
        print(f"分析数据集: {dataset_name}")
        
        # 加载数据集
        data_dict = self.load_dataset(dataset_name, dataset_path)
        if not data_dict:
            return None
        
        # 创建数据集特定的输出目录
        dataset_output_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 分析数据结构和基本统计信息
        stats = self.calculate_statistics(dataset_name, data_dict)
        
        # 绘制可视化图表 - 为SWaT数据集添加异常处理
        try:
            visualization_files = self.generate_visualizations(dataset_name, data_dict, dataset_output_dir)
        except Exception as e:
            print(f"生成{dataset_name}可视化图表出错: {e}")
            visualization_files = []
        
        # 保存统计信息
        with open(os.path.join(dataset_output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 生成HTML部分
        html_section = self.generate_dataset_html_section(dataset_name, data_dict, stats, visualization_files)
        
        return html_section
    
    def calculate_statistics(self, dataset_name, data_dict):
        """计算数据集的统计信息"""
        stats = {
            'dataset_name': dataset_name,
            'data_types': {},
            'basic_info': "",
            'feature_stats': {},
            'data_structure': []
        }
        
        # 收集数据结构信息
        for data_type, data_info in data_dict.items():
            if data_type in ['feature_names']:  # 跳过非数据项
                continue
                
            data = data_info['data']
            data_type_str = data_info['type']
            
            # 保存数据类型
            stats['data_types'][data_type] = data_type_str
            
            # 获取数据形状信息
            if isinstance(data, np.ndarray):
                shape_info = {}
                shape_info['data_type'] = data_type
                shape_info['file'] = f'({data_type_str} format)'
                shape_info['samples'] = data.shape[0] if len(data.shape) > 0 else 0
                shape_info['features'] = data.shape[1] if len(data.shape) > 1 else 1
                shape_info['other_dims'] = ', '.join([str(d) for d in data.shape[2:]]) if len(data.shape) > 2 else 'None'
                stats['data_structure'].append(shape_info)
            elif isinstance(data, pd.DataFrame):
                shape_info = {}
                shape_info['data_type'] = data_type
                shape_info['file'] = f'({data_type_str} format)'
                shape_info['samples'] = len(data)
                shape_info['features'] = len(data.columns)
                shape_info['other_dims'] = 'None'
                stats['data_structure'].append(shape_info)
        
        # 计算特征统计信息
        feature_stats = {}
        
        # 尝试从训练数据中获取特征统计
        if 'train' in data_dict:
            train_data = data_dict['train']['data']
            feature_names = data_dict.get('feature_names', [])
            
            if isinstance(train_data, np.ndarray):
                # 处理numpy数组
                if len(train_data.shape) > 1:
                    # 对于多维数据，计算每个特征的统计
                    for i in range(min(train_data.shape[1], 10)):  # 只计算前10个特征以避免信息过多
                        feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                        feature_data = train_data[:, i]
                        if len(feature_data.shape) > 1:
                            feature_data = feature_data.flatten()
                        
                        feature_stats[feature_name] = {
                            'mean': float(np.mean(feature_data)),
                            'std': float(np.std(feature_data)),
                            'min': float(np.min(feature_data)),
                            '25%': float(np.percentile(feature_data, 25)),
                            '50%': float(np.percentile(feature_data, 50)),
                            '75%': float(np.percentile(feature_data, 75)),
                            'max': float(np.max(feature_data))
                        }
            elif isinstance(train_data, pd.DataFrame):
                # 处理DataFrame
                # 只计算数值列的统计
                numeric_columns = train_data.select_dtypes(include=['number']).columns
                for col in numeric_columns[:10]:  # 只计算前10个数值列
                    feature_stats[col] = {
                        'mean': float(train_data[col].mean()),
                        'std': float(train_data[col].std()),
                        'min': float(train_data[col].min()),
                        '25%': float(train_data[col].quantile(0.25)),
                        '50%': float(train_data[col].quantile(0.5)),
                        '75%': float(train_data[col].quantile(0.75)),
                        'max': float(train_data[col].max())
                    }
        
        stats['feature_stats'] = feature_stats
        
        # 设置基本信息
        stats['basic_info'] = f"这是{dataset_name}数据集，包含以下数据部分: {', '.join(list(data_dict.keys())).replace('_', ' ')}。"
        
        return stats
    
    def generate_visualizations(self, dataset_name, data_dict, output_dir):
        """生成数据集的可视化图表"""
        visualization_files = []
        
        # 检查是否有训练数据
        if 'train' not in data_dict:
            return visualization_files
        
        train_data = data_dict['train']['data']
        feature_names = data_dict.get('feature_names', [])
        
        try:
            # 1. 特征分布直方图 (只显示前5个特征)
            self.plot_feature_distributions(train_data, feature_names, output_dir, dataset_name)
            visualization_files.append(os.path.join('..', 'dataset_analysis_reports', dataset_name, f'{dataset_name}_feature_distributions.png'))
            
            # 2. 相关性热图 (如果特征数量合适)
            if isinstance(train_data, np.ndarray) and train_data.shape[1] > 1 and train_data.shape[1] <= 50:
                self.plot_correlation_heatmap(train_data, feature_names, output_dir, dataset_name)
                visualization_files.append(os.path.join('..', 'dataset_analysis_reports', dataset_name, f'{dataset_name}_correlation_heatmap.png'))
            elif isinstance(train_data, pd.DataFrame):
                numeric_cols = train_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 1 and len(numeric_cols) <= 50:
                    self.plot_correlation_heatmap(train_data, feature_names, output_dir, dataset_name)
                    visualization_files.append(os.path.join('..', 'dataset_analysis_reports', dataset_name, f'{dataset_name}_correlation_heatmap.png'))
            
            # 3. 箱线图 (只显示前5个特征)
            self.plot_boxplots(train_data, feature_names, output_dir, dataset_name)
            visualization_files.append(os.path.join('..', 'dataset_analysis_reports', dataset_name, f'{dataset_name}_boxplots.png'))
            
            # 4. 时间序列示例 (如果数据是时间序列格式) - 为SWaT添加额外的错误处理
            if 'test' in data_dict and 'test_labels' in data_dict:
                try:
                    self.plot_time_series_example(data_dict, output_dir, dataset_name)
                    visualization_files.append(os.path.join('..', 'dataset_analysis_reports', dataset_name, f'{dataset_name}_time_series_example.png'))
                except Exception as e:
                    print(f"生成{dataset_name}时间序列图出错: {e}")
                
                try:
                    # 5. 标签分布
                    self.plot_label_distribution(data_dict, output_dir, dataset_name)
                    visualization_files.append(os.path.join('..', 'dataset_analysis_reports', dataset_name, f'{dataset_name}_label_distribution.png'))
                except Exception as e:
                    print(f"生成{dataset_name}标签分布图出错: {e}")
            
        except Exception as e:
            print(f"生成{dataset_name}可视化图表出错: {e}")
        
        return visualization_files
    
    def plot_feature_distributions(self, data, feature_names, output_dir, dataset_name):
        """绘制特征分布直方图"""
        plt.figure(figsize=(15, 10))
        
        try:
            if isinstance(data, np.ndarray):
                # 处理numpy数组
                num_features = min(data.shape[1], 5)  # 只显示前5个特征
                for i in range(num_features):
                    ax = plt.subplot(math.ceil(num_features/2), 2, i+1)
                    feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                    
                    # 处理多维数据
                    feature_data = data[:, i]
                    if len(feature_data.shape) > 1:
                        feature_data = feature_data.flatten()
                    
                    # 标准化数据以便可视化
                    scaler = StandardScaler()
                    feature_data_scaled = scaler.fit_transform(feature_data.reshape(-1, 1)).flatten()
                    
                    sns.histplot(feature_data_scaled, bins=50, kde=True, ax=ax)
                    ax.set_title(f'{feature_name} (标准化后)')
                    ax.set_xlabel('值')
                    ax.set_ylabel('频率')
            elif isinstance(data, pd.DataFrame):
                # 处理DataFrame
                numeric_cols = data.select_dtypes(include=['number']).columns
                num_features = min(len(numeric_cols), 5)  # 只显示前5个数值特征
                
                for i, col in enumerate(numeric_cols[:num_features]):
                    ax = plt.subplot(math.ceil(num_features/2), 2, i+1)
                    
                    # 标准化数据以便可视化
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data[[col]]).flatten()
                    
                    sns.histplot(data_scaled, bins=50, kde=True, ax=ax)
                    ax.set_title(f'{col} (标准化后)')
                    ax.set_xlabel('值')
                    ax.set_ylabel('频率')
        except Exception as e:
            print(f"绘制特征分布图出错: {e}")
            plt.title('特征分布图 (无法正常生成)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self, data, feature_names, output_dir, dataset_name):
        """绘制特征相关性热图"""
        plt.figure(figsize=(12, 10))
        
        try:
            if isinstance(data, np.ndarray):
                # 计算相关性矩阵
                if len(data.shape) > 2:
                    # 对于多维数据，取第一个时间步或展平
                    data_2d = data[:, :, 0] if data.shape[2] > 0 else data.reshape(data.shape[0], -1)
                else:
                    data_2d = data
                
                corr_matrix = np.corrcoef(data_2d, rowvar=False)
                # 使用前n个特征名
                display_names = feature_names[:data_2d.shape[1]] if len(feature_names) >= data_2d.shape[1] else [f'feature_{i}' for i in range(data_2d.shape[1])]
            elif isinstance(data, pd.DataFrame):
                # 选择数值列计算相关性
                numeric_cols = data.select_dtypes(include=['number']).columns
                corr_matrix = data[numeric_cols].corr().values
                display_names = list(numeric_cols)
            
            # 绘制热图
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                        xticklabels=display_names, yticklabels=display_names, 
                        square=True, linewidths=.5, cbar_kws={'shrink': .8})
            
            plt.title('特征相关性热图')
            plt.xticks(rotation=45, ha='right')
        except Exception as e:
            print(f"绘制相关性热图出错: {e}")
            plt.title('相关性热图 (无法正常生成)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_boxplots(self, data, feature_names, output_dir, dataset_name):
        """绘制特征箱线图"""
        plt.figure(figsize=(15, 8))
        
        # 准备数据用于箱线图
        boxplot_data = []
        labels = []
        
        try:
            if isinstance(data, np.ndarray):
                # 处理numpy数组
                num_features = min(data.shape[1], 5)  # 只显示前5个特征
                for i in range(num_features):
                    feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                    
                    # 处理多维数据
                    feature_data = data[:, i]
                    if len(feature_data.shape) > 1:
                        feature_data = feature_data.flatten()
                    
                    # 标准化数据
                    scaler = StandardScaler()
                    feature_data_scaled = scaler.fit_transform(feature_data.reshape(-1, 1)).flatten()
                    
                    boxplot_data.append(feature_data_scaled)
                    labels.append(feature_name)
            elif isinstance(data, pd.DataFrame):
                # 处理DataFrame
                numeric_cols = data.select_dtypes(include=['number']).columns
                num_features = min(len(numeric_cols), 5)  # 只显示前5个数值特征
                
                for col in numeric_cols[:num_features]:
                    # 标准化数据
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data[[col]]).flatten()
                    
                    boxplot_data.append(data_scaled)
                    labels.append(col)
            
            # 绘制箱线图
            sns.boxplot(data=boxplot_data)
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.title('特征分布箱线图 (标准化后)')
            plt.ylabel('标准化值')
        except Exception as e:
            print(f"绘制箱线图出错: {e}")
            plt.boxplot([[1, 2, 3, 4, 5]])
            plt.title('箱线图 (无法正常生成)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_boxplots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series_example(self, data_dict, output_dir, dataset_name):
        """绘制时间序列示例"""
        test_data = data_dict['test']['data']
        test_labels = data_dict['test_labels']['data']
        feature_names = data_dict.get('feature_names', [])
        
        plt.figure(figsize=(15, 10))
        
        # 选择一个样本进行可视化
        sample_idx = 0
        
        try:
            if isinstance(test_data, np.ndarray):
                # 处理numpy数组
                # 对于三维数据，选择第一个样本的前5个特征
                if len(test_data.shape) == 3:
                    try:
                        sample_data = test_data[sample_idx, :, :5]  # 第一个样本的前5个时间序列
                        num_features = sample_data.shape[0]
                        time_steps = sample_data.shape[1]
                        
                        for i in range(num_features):
                            ax = plt.subplot(num_features + 1, 1, i + 1)
                            feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                            ax.plot(range(time_steps), sample_data[i])
                            ax.set_title(f'{feature_name}')
                            ax.set_ylabel('值')
                            if i < num_features - 1:
                                ax.set_xticklabels([])
                        
                        # 在最后添加标签
                        ax = plt.subplot(num_features + 1, 1, num_features + 1)
                        if len(test_labels.shape) > 1:
                            ax.plot(range(time_steps), test_labels[sample_idx, :])
                        else:
                            ax.plot(range(min(time_steps, len(test_labels))), test_labels[:min(time_steps, len(test_labels))])
                        ax.set_title('异常标签')
                        ax.set_xlabel('时间步')
                        ax.set_ylabel('异常标记')
                    except Exception as e:
                        print(f"绘制时间序列出错: {e}")
                        # 创建一个简单的图形以避免返回空
                        plt.plot([0, 1, 2, 3], [0, 1, 0, 1])
                        plt.title('时间序列示例 (数据格式不支持详细可视化)')
        except Exception as e:
            print(f"绘制时间序列示例出错: {e}")
            plt.title('时间序列示例 (无法正常生成)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_time_series_example.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_label_distribution(self, data_dict, output_dir, dataset_name):
        """绘制标签分布"""
        test_labels = data_dict['test_labels']['data']
        
        plt.figure(figsize=(10, 6))
        
        try:
            # 将标签展平
            if isinstance(test_labels, np.ndarray):
                labels_flat = test_labels.flatten()
            elif isinstance(test_labels, pd.DataFrame):
                labels_flat = test_labels.values.flatten()
            
            try:
                # 计算正常和异常的数量
                normal_count = int(np.sum(labels_flat == 0))
                anomaly_count = int(np.sum(labels_flat == 1))
                
                # 绘制饼图
                plt.pie([normal_count, anomaly_count], labels=['正常数据', '异常数据'], autopct='%1.1f%%',
                        shadow=True, startangle=90, colors=['#4CAF50', '#f44336'])
                plt.axis('equal')  # 保证饼图是圆的
                plt.title('数据集标签分布')
            except Exception as e:
                print(f"绘制标签分布出错: {e}")
                # 创建一个简单的图形以避免返回空
                plt.bar(['正常', '异常'], [100, 10])
                plt.title('标签分布 (无法准确计算分布)')
        except Exception as e:
            print(f"绘制标签分布示例出错: {e}")
            plt.title('标签分布 (无法正常生成)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_label_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_dataset_html_section(self, dataset_name, data_dict, stats, visualization_files):
        """生成数据集的HTML部分"""
        # 生成数据结构行
        data_structure_rows = []
        for struct in stats['data_structure']:
            row = f"""
                        <tr>
                            <td>{struct['data_type']}</td>
                            <td>{struct['file']}</td>
                            <td>{struct['samples']}</td>
                            <td>{struct['features']}</td>
                            <td>{struct['other_dims']}</td>
                        </tr>"""
            data_structure_rows.append(row)
        
        # 生成统计行
        stats_rows = []
        for feature, feature_stats in stats['feature_stats'].items():
            row = f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{feature_stats['mean']:.4f}</td>
                            <td>{feature_stats['std']:.4f}</td>
                            <td>{feature_stats['min']:.4f}</td>
                            <td>{feature_stats['25%']:.4f}</td>
                            <td>{feature_stats['50%']:.4f}</td>
                            <td>{feature_stats['75%']:.4f}</td>
                            <td>{feature_stats['max']:.4f}</td>
                        </tr>"""
            stats_rows.append(row)
        
        # 生成数据样本
        sample_rows = []
        try:
            if 'train' in data_dict:
                train_data = data_dict['train']['data']
                feature_names = data_dict.get('feature_names', [])
                
                # 获取样本数据
                if isinstance(train_data, np.ndarray):
                    # 取前5个样本，前5个特征
                    num_samples = min(5, train_data.shape[0])
                    num_features = min(5, train_data.shape[1] if len(train_data.shape) > 1 else 1)
                    
                    # 创建表头
                    headers = ['样本索引'] + [feature_names[i] if i < len(feature_names) else f'feature_{i}' for i in range(num_features)]
                    sample_rows.append('<tr>' + ''.join([f'<th>{h}</th>' for h in headers]) + '</tr>')
                    
                    # 添加数据行
                    for i in range(num_samples):
                        row_data = [f'<td>{i}</td>']
                        for j in range(num_features):
                            if len(train_data.shape) > 2:
                                # 多维数据，取第一个时间步的值
                                val = train_data[i, j, 0] if train_data.shape[2] > 0 else train_data[i, j]
                            else:
                                val = train_data[i, j]
                            row_data.append(f'<td>{val:.4f}</td>')
                        sample_rows.append('<tr>' + ''.join(row_data) + '</tr>')
                elif isinstance(train_data, pd.DataFrame):
                    # 取前5个样本，前5个列
                    num_samples = min(5, len(train_data))
                    num_cols = min(5, len(train_data.columns))
                    
                    # 创建表头
                    headers = ['样本索引'] + list(train_data.columns[:num_cols])
                    sample_rows.append('<tr>' + ''.join([f'<th>{h}</th>' for h in headers]) + '</tr>')
                    
                    # 添加数据行
                    for i in range(num_samples):
                        row_data = [f'<td>{i}</td>']
                        for col in train_data.columns[:num_cols]:
                            val = train_data.iloc[i][col]
                            if isinstance(val, (int, float)):
                                row_data.append(f'<td>{val:.4f}</td>')
                            else:
                                row_data.append(f'<td>{str(val)[:20]}</td>')  # 限制字符串长度
                        sample_rows.append('<tr>' + ''.join(row_data) + '</tr>')
        except Exception as e:
            print(f"生成{dataset_name}数据样本出错: {e}")
            sample_rows = ['<tr><td colspan="6">无法生成数据样本</td></tr>']
        
        # 生成可视化图像
        visualization_images = []
        for img_path in visualization_files:
            img_name = os.path.basename(img_path).replace(f'{dataset_name}_', '').replace('.png', '')
            img_title = img_name.replace('_', ' ').capitalize()
            img_html = f"""
                    <div class="image-item">
                        <h4>{img_title}</h4>
                        <img src="{img_path}" alt="{img_title}">
                    </div>"""
            visualization_images.append(img_html)
        
        # 生成数据情况分析
        data_analysis = self.generate_data_analysis(data_dict, dataset_name)
        
        # 生成数据处理建议
        data_recommendations = self.generate_data_recommendations(data_dict, dataset_name)
        
        # 生成完整的HTML部分
        html_section = self.dataset_section_template.format(
            dataset_name=dataset_name,
            basic_info=stats['basic_info'],
            data_structure_rows=''.join(data_structure_rows),
            stats_rows=''.join(stats_rows),
            sample_rows=''.join(sample_rows),
            data_analysis=data_analysis,
            visualization_images=''.join(visualization_images),
            data_recommendations=data_recommendations
        )
        
        return html_section
    
    def generate_summary_statistics(self):
        """生成所有数据集的汇总统计信息"""
        summary_html = f"""
            <div class="stats-table-container">
                <table class="stats-table">
                    <tr>
                        <th>数据集</th>
                        <th>数据类型</th>
                        <th>训练样本数</th>
                        <th>测试样本数</th>
                        <th>特征数</th>
                        <th>是否有标签</th>
                    </tr>"""
        
        for dataset_name, dataset_path in self.dataset_paths.items():
            if os.path.exists(dataset_path):
                try:
                    # 加载数据集信息
                    data_dict = self.load_dataset(dataset_name, dataset_path)
                    
                    # 获取训练和测试样本数
                    train_samples = 0
                    test_samples = 0
                    num_features = 0
                    has_labels = 'test_labels' in data_dict
                    
                    if 'train' in data_dict:
                        train_data = data_dict['train']['data']
                        if isinstance(train_data, np.ndarray):
                            train_samples = train_data.shape[0]
                            num_features = train_data.shape[1] if len(train_data.shape) > 1 else 1
                        elif isinstance(train_data, pd.DataFrame):
                            train_samples = len(train_data)
                            num_features = len(train_data.columns)
                    
                    if 'test' in data_dict:
                        test_data = data_dict['test']['data']
                        if isinstance(test_data, np.ndarray):
                            test_samples = test_data.shape[0]
                        elif isinstance(test_data, pd.DataFrame):
                            test_samples = len(test_data)
                    
                    # 获取数据类型
                    data_types = ', '.join(list(data_dict.get('data_types', {}).values()))
                    
                    summary_html += f"""
                    <tr>
                        <td>{dataset_name}</td>
                        <td>{data_types}</td>
                        <td>{train_samples}</td>
                        <td>{test_samples}</td>
                        <td>{num_features}</td>
                        <td>{'是' if has_labels else '否'}</td>
                    </tr>"""
                except Exception as e:
                    print(f"生成{dataset_name}汇总信息出错: {e}")
                    summary_html += f"""
                    <tr>
                        <td>{dataset_name}</td>
                        <td>加载错误</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                        <td>-</td>
                    </tr>"""
        
        summary_html += f"""
                </table>
            </div>"""
        
        return summary_html
    
    def analyze_dataset(self, dataset_name, dataset_path):
        """分析单个数据集"""
        print(f"分析数据集: {dataset_name}")
        
        try:
            # 加载数据集
            data_dict = self.load_dataset(dataset_name, dataset_path)
            
            if not data_dict:
                print(f"无法加载{dataset_name}数据集")
                return None
            
            # 计算统计信息 - 修复参数顺序
            stats = self.calculate_statistics(dataset_name, data_dict)
            
            # 创建数据集特定的输出目录
            dataset_output_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # 生成可视化图表
            visualization_files = []
            
            # 绘制特征分布图 - 修复参数顺序
            self.plot_feature_distributions(data_dict['train']['data'], data_dict.get('feature_names', []), dataset_output_dir, dataset_name)
            visualization_files.append(os.path.join(dataset_name, f'{dataset_name}_feature_distributions.png'))
            
            # 绘制相关性热图 - 修复参数顺序
            self.plot_correlation_heatmap(data_dict['train']['data'], data_dict.get('feature_names', []), dataset_output_dir, dataset_name)
            visualization_files.append(os.path.join(dataset_name, f'{dataset_name}_correlation_heatmap.png'))
            
            # 绘制箱线图 - 修复参数顺序
            self.plot_boxplots(data_dict['train']['data'], data_dict.get('feature_names', []), dataset_output_dir, dataset_name)
            visualization_files.append(os.path.join(dataset_name, f'{dataset_name}_boxplots.png'))
            
            # 绘制时间序列示例（如果适用）
            if dataset_name in ['MSL', 'SMAP', 'SMD'] and 'test' in data_dict and 'test_labels' in data_dict:
                self.plot_time_series_example(data_dict, dataset_output_dir, dataset_name)
                visualization_files.append(os.path.join(dataset_name, f'{dataset_name}_time_series_example.png'))
            
            # 绘制标签分布
            if 'test_labels' in data_dict:
                self.plot_label_distribution(data_dict, dataset_output_dir, dataset_name)
                visualization_files.append(os.path.join(dataset_name, f'{dataset_name}_label_distribution.png'))
            
            # 生成HTML部分
            html_section = self.generate_dataset_html_section(dataset_name, data_dict, stats, visualization_files)
            
            return html_section
        except Exception as e:
            print(f"分析{dataset_name}数据集时出错: {e}")
            return None
    
    def analyze_all_datasets(self):
        """分析所有数据集并生成综合报告"""
        print("开始分析所有数据集...")
        
        # 生成每个数据集的HTML部分
        dataset_sections = []
        
        for dataset_name, dataset_path in self.dataset_paths.items():
            if os.path.exists(dataset_path):
                html_section = self.analyze_dataset(dataset_name, dataset_path)
                if html_section:
                    dataset_sections.append(html_section)
            else:
                print(f"数据集路径不存在: {dataset_path}")
        
        # 生成汇总统计信息
        summary_stats = self.generate_summary_statistics()
        
        # 生成数据集名称列表
        dataset_names = ', '.join([name for name, path in self.dataset_paths.items() if os.path.exists(path)])
        
        # 填充HTML报告
        try:
            # 使用字符串连接而不是.format()方法来避免CSS大括号解析问题
            # 先将模板中的格式化标记替换为特殊字符串
            temp_html = self.html_template
            temp_html = temp_html.replace('{dataset_names}', '___DATASET_NAMES___')
            temp_html = temp_html.replace('{summary_stats}', '___SUMMARY_STATS___')
            temp_html = temp_html.replace('{dataset_sections}', '___DATASET_SECTIONS___')
            
            # 然后替换这些特殊字符串为实际内容
            final_html = temp_html.replace('___DATASET_NAMES___', dataset_names)
            final_html = final_html.replace('___SUMMARY_STATS___', summary_stats)
            final_html = final_html.replace('___DATASET_SECTIONS___', ''.join(dataset_sections))
        except Exception as e:
            print(f"HTML报告生成出错: {e}")
            # 创建一个简化版的HTML报告，避免因模板问题导致整个过程失败
            final_html = '''
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <title>数据集分析报告</title>
                <style>
                    body { font-family: Arial, SimHei, sans-serif; margin: 20px; }
                    h1, h2 { color: #333; }
                </style>
            </head>
            <body>
                <h1>数据集分析报告</h1>
                <p>报告生成过程中遇到问题，但已分析的数据集包括: ''' + dataset_names + '''</p>
                <div>''' + summary_stats + '''</div>
                <div>''' + ''.join(dataset_sections) + '''</div>
            </body>
            </html>'''
        
        # 保存最终的HTML报告
        report_path = os.path.join(self.output_dir, 'dataset_analysis_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"数据分析报告已生成: {report_path}")
        
        return report_path

if __name__ == "__main__":
    # 获取当前目录作为根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建分析器实例并运行
    analyzer = DatasetAnalyzer(root_dir)
    report_path = analyzer.analyze_all_datasets()
    
    print(f"分析完成! 请查看报告: {report_path}")