#!/bin/bash

# 激活lzytest环境
source activate lzytest

# 设置PYTHONPATH
export PYTHONPATH=/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main


python anomaly_detection/save_chunked_data.py \
    --data 'MSL' \
    --batch_size 128 \
    --task_name 'anomaly_detection' \
    --root_path "/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main/datasets/MSL/" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/MSL/" \
    --num_vars 55 \
    --sample_size 1000

python anomaly_detection/revin_data.py \
    --root_path "anomaly_detection/data/MSL/" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/MSL/revin_data/" \
    --num_vars 55

for seed in 47
do
python anomaly_detection/train_vqvae.py \
    --config_path anomaly_detection/scripts/msl.json \
    --model_init_num_gpus 0 \
    --data_init_cpu_or_gpu cpu \
    --save_path "anomaly_detection/saved_models/MSL/" \
    --base_path "anomaly_detection/data/MSL/revin_data/" \
    --batchsize 200 \
    --seed $seed
done

seed=47
python anomaly_detection/detect_anomaly.py \
    --dataset "MSL"\
    --trained_vqvae_model_path "anomaly_detection/saved_models/MSL/CD64_CW1024_CF4_BS4096_ITR3000_seed47/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --base_path "anomaly_detection/data/MSL/revin_data"\
    --labels_path "anomaly_detection/data/MSL"\
    --anomaly_ratio 2 \
    --gpu 0 \
    --num_vars 55 \
    --seq_len 100