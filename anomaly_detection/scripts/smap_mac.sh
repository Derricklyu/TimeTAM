#!/bin/bash

# 解决OMP库冲突问题
export KMP_DUPLICATE_LIB_OK=TRUE

# 确保中文显示正常
export LC_ALL=zh_CN.UTF-8
export LANG=zh_CN.UTF-8

# 设置数据集路径
data_root_path="/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/datasets/SMAP"
save_data_path="/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/anomaly_detection/data/SMAP/"
revin_data_path="/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/anomaly_detection/data/SMAP/revin_data/"
saved_model_path="/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/anomaly_detection/saved_models/SMAP/"
report_path="/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/smap_anomaly_detection_report/"

# 创建必要的目录
mkdir -p "$save_data_path"
mkdir -p "$revin_data_path"
mkdir -p "$saved_model_path"
mkdir -p "$report_path"

# 步骤1: 数据预处理
echo "步骤1: 数据预处理..."
python save_chunked_data.py \
    --data 'SMAP' \
    --batch_size 128 \
    --task_name 'anomaly_detection' \
    --root_path "$data_root_path" \
    --seq_len 100 \
    --save_path "$save_data_path" \
    --num_vars 25 \
    --gpu 0

# 步骤2: 应用RevIN转换
echo "步骤2: 应用RevIN转换..."
python revin_data.py \
    --root_path "$save_data_path" \
    --seq_len 100 \
    --save_path "$revin_data_path" \
    --num_vars 25

# 步骤3: 训练VQ-VAE模型
# 在Mac上使用mps后端代替cuda
echo "步骤3: 训练VQ-VAE模型..."
seed=47
python train_vqvae.py \
    --config_path scripts/smap.json \
    --model_init_num_gpus 0 \
    --data_init_cpu_or_gpu cpu \
    --comet_log \
    --comet_tag mac_gpu \
    --comet_name smap_mac \
    --save_path "$saved_model_path" \
    --base_path "$revin_data_path" \
    --batchsize 2048 \
    --seed $seed

# 步骤4: 检测异常并生成报告
echo "步骤4: 检测异常并生成报告..."
# 首先需要找到训练好的模型的确切路径
model_dir=$(ls -d "$saved_model_path"*"seed"$seed | head -1)

# 检测异常并生成报告
python run_smap_anomaly_detection.py \
    --dataset "SMAP" \
    --trained_vqvae_model_path "$model_dir/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --base_path "$revin_data_path" \
    --labels_path "$save_data_path" \
    --anomaly_ratio 1 \
    --num_vars 25 \
    --seq_len 100 \
    --report_path "$report_path"