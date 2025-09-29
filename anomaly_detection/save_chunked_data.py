import os
# 解决OMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import argparse
import numpy as np
import data_provider.data_factory as data_factory


def save_data(data_set, args, save_labels=False):
    all_data = []
    all_labels = []
    for i, (example, label) in enumerate(data_set):
        # 如果设置了采样大小，并且已经达到了采样数量，就停止处理
        if hasattr(args, 'sample_size') and args.sample_size is not None and i >= args.sample_size:
            break
            
        if example.shape != (args.seq_len, args.num_vars):  # do this becuase their loaders have the train set come with the test_labels (which are longer)
            break

        all_data.append(example.reshape(1, example.shape[0], example.shape[-1]))
        if save_labels:
            all_labels.append(label.reshape(1, label.shape[0]))

    if not all_data:
        return np.array([]), np.array([]) if save_labels else None
        
    if save_labels:
        return np.concatenate(all_data, axis=0), np.concatenate(all_labels, axis=0)
    else:
        return np.concatenate(all_data, axis=0), None


def process_data(args):
    # 从数据集获取特征数量
    if args.data == 'SMAP':
        # 加载一小部分数据来获取特征数量
        sample_data = np.load(os.path.join(args.root_path, 'SMAP_train.npy'))
        args.num_vars = sample_data.shape[1]
        print(f"自动检测到特征数量: {args.num_vars}")
        
    train_data_set, train_data_loader = data_factory.data_provider(args, 'train')
    all_train, _ = save_data(train_data_set, args, save_labels=False)

    test_data_set, test_data_loader = data_factory.data_provider(args, 'test')
    all_test, all_test_labels = save_data(test_data_set, args, save_labels=True)

    print(f"训练数据形状: {all_train.shape}")
    print(f"测试数据形状: {all_test.shape}")
    
    # 检查数据是否为空
    if len(all_train) == 0:
        print("警告: 没有加载到训练数据，请检查数据路径或参数设置")
        return
        
    if len(all_test) == 0:
        print("警告: 没有加载到测试数据，请检查数据路径或参数设置")
        return

    if all_test_labels is not None:
        print(f"测试标签形状: {all_test_labels.shape}")
    
    # 验证形状
    if len(all_train.shape) >= 2 and len(all_train.shape) >= 3:
        if all_train.shape[1] != args.seq_len or all_train.shape[2] != args.num_vars:
            print(f'训练数据形状不正确: {all_train.shape}, 期望的序列长度: {args.seq_len}, 特征数量: {args.num_vars}')
            # 不使用pdb，继续执行以保存可用数据

    if len(all_test.shape) >= 2 and len(all_test.shape) >= 3:
        if all_test.shape[1] != args.seq_len or all_test.shape[2] != args.num_vars:
            print(f'测试数据形状不正确: {all_test.shape}, 期望的序列长度: {args.seq_len}, 特征数量: {args.num_vars}')
            # 不使用pdb，继续执行以保存可用数据

    if all_test_labels is not None and len(all_test_labels.shape) >= 2:
        if all_test_labels.shape[1] != args.seq_len:
            print(f'测试标签形状不正确: {all_test_labels.shape}, 期望的序列长度: {args.seq_len}')
            # 不使用pdb，继续执行以保存可用数据

    if all_test.shape[0] != all_test_labels.shape[0]:
        print('something funky with test sizes ')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    np.save(args.save_path + 'train_data_processed.npy', all_train, allow_pickle=True)
    np.save(args.save_path + 'test_data_processed.npy', all_test, allow_pickle=True)
    np.save(args.save_path + 'test_labels_processed.npy', all_test_labels, allow_pickle=True)

    print('FINISHED STEP 1')
    print('---------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        required=False, default='',
                        help='dataset name')
    parser.add_argument('--batch_size', type=int,
                        required=True,
                        help='batchsize')

    parser.add_argument('--task_name', type=str,
                        required=True,
                        help='name of the task')

    parser.add_argument('--root_path', type=str,
                        required=True,
                        help='path to the data')

    parser.add_argument('--save_path', type=str,
                        required=True,
                        help='path to save the data')

    parser.add_argument('--seq_len', type=int,
                        required=True,
                        help='window size to reconstruct')

    parser.add_argument('--num_workers', type=int,
                        required=False, default=10,
                        help='number of workers')

    parser.add_argument('--num_vars', type=int,
                        required=False,
                        help='number of sensors')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--sample_size', type=int, default=None, help='number of samples to process, None for all samples')

    args = parser.parse_args()

    if args.data == 'MSL' or args.data == 'PSM' or args.data == 'SMAP' or args.data == 'SMD' or args.data == 'SWAT':
        process_data(args)