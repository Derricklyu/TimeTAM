# 解决OMP库冲突问题 - 在导入任何模块之前就设置这个环境变量
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import argparse
import numpy as np
import torch
from layers.RevIN import RevIN


class ExtractData:
    def __init__(self, data, device, num_features):
        self.revin_layer = RevIN(num_features=num_features, affine=False, subtract_last=False)
        self.data = data
        self.device = device

    def one_loop(self, x):
        x_in_revin_space = []
        # data should have dimension batch, time, sensors (features)
        batch_x = torch.tensor(x)
        batch_x = batch_x.float().to(self.device)
        
        # 打印输入数据的基本信息
        print(f"[RevIN处理] 输入数据形状: {batch_x.shape}, 数据类型: {batch_x.dtype}")
        print(f"[RevIN处理] 输入数据统计 - 均值: {torch.mean(batch_x):.4f}, 标准差: {torch.std(batch_x):.4f}")

        # data going into revin should have dim:[bs x seq_len x nvars]
        print("[RevIN处理] 开始执行归一化操作...")
        x_norm = self.revin_layer(batch_x, "norm")
        print(f"[RevIN处理] 归一化后数据统计 - 均值: {torch.mean(x_norm):.4f}, 标准差: {torch.std(x_norm):.4f}")
        
        x_in_revin_space.append(np.array(x_norm.detach().cpu()))
        x_in_revin_space_arr = np.concatenate(x_in_revin_space, axis=0)

        print(f"[RevIN处理] 输出数据形状: {x_in_revin_space_arr.shape}")
        return x_in_revin_space_arr

    def extract_data(self):
        print('[RevIN处理开始] 准备对数据进行RevIN转换')
        print(f'[RevIN处理开始] 原始数据形状: {self.data.shape}')
        # These have dimension [bs, ntime, nvars]
        data_in_revin_space_arr = self.one_loop(self.data)
        print('[RevIN处理完成] 数据转换已完成')
        return data_in_revin_space_arr


def do_data(data, device, num_feature):
    # data should have dimension: [bs x seq_len x nvars] --> so need to swap axis 1 & 2
    print(f'[数据处理流程] 开始处理数据，特征数量: {num_feature}，设备: {device}')
    Exp = ExtractData
    exp = Exp(data, device, num_feature)  # set experiments
    result = exp.extract_data()
    print(f'[数据处理流程] 数据处理完成，结果形状: {result.shape}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        required=True,
                        help='path to the data')

    parser.add_argument('--save_path', type=str,
                        required=True,
                        help='path to save the data')

    parser.add_argument('--seq_len', type=int,
                        required=True,
                        help='window size to reconstruct')

    parser.add_argument('--num_vars', type=int,
                        required=False,
                        help='number of sensors')

    parser.add_argument('--gpu', type=int, default=1, help='gpu')

    args = parser.parse_args()

    # 检查CUDA是否可用，如果不可用则使用CPU
    try:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print(f"[警告] CUDA不可用，使用CPU设备")
    except:
        device = torch.device('cpu')
        print(f"[警告] CUDA配置失败，使用CPU设备")

    print(f"[主流程] 使用设备: {device}")

    base_path = args.root_path
    base_save_path = args.save_path
    num_features = args.num_vars

    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    else:
        print('The data path already exists')
    #pdb.set_trace()

    # do train first
    data_train = np.load(base_path + "train_data_processed.npy", allow_pickle=True)
    print('[主流程] 加载训练数据完成')
    print(f'[主流程] 初始训练数据形状: {data_train.shape}')
    print(f'[主流程] 开始对训练数据执行RevIN处理...')
    data_train_revined = do_data(data_train, device, num_feature=num_features)
    print('[主流程] 训练数据RevIN处理完成')

    data_test = np.load(base_path + "test_data_processed.npy", allow_pickle=True)
    print('[主流程] 加载测试数据完成')
    print(f'[主流程] 初始测试数据形状: {data_test.shape}')
    print(f'[主流程] 开始对测试数据执行RevIN处理...')
    data_test_revined = do_data(data_test, device, num_feature=num_features)
    print('[主流程] 测试数据RevIN处理完成')

    print(f'[主流程] RevIN处理后的训练数据形状: {data_train_revined.shape}')
    print(f'[主流程] RevIN处理后的测试数据形状: {data_test_revined.shape}')

    if data_train_revined.shape[1] != args.seq_len or data_train_revined.shape[2] != args.num_vars:
        print('Train has shape problem')
        #pdb.set_trace()

    if data_test_revined.shape[1] != args.seq_len or data_test_revined.shape[2] != args.num_vars:
        print('Test has shape problem')
        #pdb.set_trace()

    print(f'[主流程] 准备保存RevIN处理后的数据到: {base_save_path}')
    np.save(base_save_path + '/train.npy', data_train_revined, allow_pickle=True)
    np.save(base_save_path + '/test.npy', data_test_revined, allow_pickle=True)
    print('[主流程] 数据保存完成')

    print('FINISHED STEP 2')
    print('---------------')