# 激活lzytest环境
source activate lzytest

# 设置PYTHONPATH
export PYTHONPATH=/Users/derrick_lyu/Desktop/课题研究/实验/TOTEM-main

seed=47
python anomaly_detection/detect_anomaly.py \
    --dataset "MSL"\
    --trained_vqvae_model_path "anomaly_detection/saved_models/MSL/CD64_CW1024_CF4_BS4096_ITR3000_seed47/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --base_path "anomaly_detection/data/MSL/revin_data"\
    --labels_path "anomaly_detection/data/MSL"\
    --anomaly_ratio 2 \
    --num_vars 55 \
    --seq_len 100 \
    --use_gpu True