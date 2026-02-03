from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

def read_all_basin(config, seg):
    path = config['path'] + '/all_basin.txt'

    with open(path, 'r', encoding='utf-8') as f:
        all_basin = [eval(line.strip()) for line in f if line.strip()]

    n = len(all_basin)
    ed = [0] + [int(n * 0.125 * i) for i in range(1, 8)]
    ed.append(n)
    # print(ed)
    all_basin = all_basin[ed[seg - 1]: ed[seg]]
    return all_basin


class SingleBasinDataset(Dataset):
    def __init__(self, basin_id,
                 area_id, data_dir,
                 train_rate, val_rate,
                 seq_len, tar_len,
                 eps=1e-5, mode='train'):
        """
        Args:
            basin_id (str): 盆地编号，like "01013500"
            area_id (str): 区域编号，like "01"
            data_dir(str): 根目录
            seq_len (int): 输入序列长度 (encoder or decoder length)
            tar_len (int): 预测长度
            mode (str): 'train', 'val', or 'test' 用于划分数据
        """
        self.seq_len = seq_len
        self.target_len = tar_len

        'dataset/processed_data/01/01013500.csv'
        file_path = data_dir + f"/{area_id}/{basin_id}.csv"

        # 读取数据 (这里假设数据已经归一化好了，如果没有，需要在 __getitem__ 前处理)
        df = pd.read_csv(file_path)

        data = df.iloc[:, 1:].values  # 转为 numpy 数组

        # 2. 划分训练/验证/测试集 (简单的按时间切分)
        n = len(data)
        train_end = int(n * train_rate)
        val_end = int(n * val_rate) + train_end
        train_x = data[:train_end, :-1]
        train_y = data[:train_end, -1:]

        self.x_mean = np.nanmean(train_x, axis=0, keepdims=True)
        self.x_std = np.nanstd(train_x, axis=0, keepdims=True)
        self.y_mean = np.nanmean(train_y, axis=0, keepdims=True)
        self.y_std = np.nanstd(train_y, axis=0, keepdims=True)

        if mode == 'train':
            self.x_data = data[:train_end, :-1]
            self.y_data = data[:train_end, -1:]
        elif mode == 'val':
            self.x_data = data[train_end:val_end, :-1]
            self.y_data = data[train_end:val_end, -1:]
        else:
            self.x_data = data[val_end:, :-1]
            self.y_data = data[val_end:, -1:]

        self.x_data = (self.x_data - self.x_mean) / (self.x_std + eps)
        self.y_data = (self.y_data - self.y_mean) / (self.y_std + eps)
        self.length = len(self.x_data) - self.seq_len + 1

        # 3. 计算合法的样本数量
        # 因为要做滑动窗口，最后一段不足 seq_len + target_len 的数据不能用

        self.valid_indices = []

        total_samples = len(self.x_data) - self.seq_len + 1
        self.length = len(self.x_data) - self.seq_len + 1
        # print(total_samples)
        for i in range(total_samples):
            full_y_window = self.y_data[i: i + self.seq_len]
            first_token = full_y_window[0]

            if np.isnan(first_token).any():
                continue
            target_y = full_y_window[-self.target_len:]
            if np.isnan(target_y).any():
                continue

            self.valid_indices.append(i)


        # print(len(self.valid_indices))
        # print('\n')

    # -----------------------------------------------------------
    # 【核心修改结束】
    # -----------------------------------------------------------

    def __len__(self):
        # 如果数据太短，返回0
        # return self.length
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        # X: 从 idx 开始，取 seq_len 长
        x = self.x_data[idx: idx + self.seq_len].copy()
        y = self.y_data[idx: idx + self.seq_len].copy()

        y_val = y[-self.target_len:].copy()
        y[-self.target_len:] = 0

        # 转为 Tensor
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(y_val)


def get_dataloader(config, basin_id, area_id, mode='train'):
    """

    :param config: config['data']
    :param basin_id: 盆地id
    :param area_id: 区域id
    :param mode: 模式 ['train', 'val', 'test']
    :return: 返回Dataloader
    """
    dataset = SingleBasinDataset(
        basin_id=basin_id,
        area_id=area_id,
        data_dir=config['path'],
        seq_len=config['seq_len'],
        tar_len=config['tar_len'],
        train_rate=config['split_rate']['train_rate'],
        val_rate=config['split_rate']['valid_rate'],
        mode=mode
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(mode == 'train'),  # 训练时打乱，验证测试时不打乱
        num_workers=config['num_workers']
    )

    return loader
