from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import torch
import yaml

def read_all_basin(config):
    path = config['path'] + '/all_basin.txt'

    with open(path, 'r', encoding='utf-8') as f:
        all_basin = [eval(line.strip()) for line in f if line.strip()]
    return all_basin


class SingleBasinDataset(Dataset):
    def __init__(self, basin_id,
                 area_id, data_dir,
                 config,
                 x_mean_static, x_std_static,
                 seq_len, tar_len,
                 eps=1e-5, mode='train'):
        """
        Args:
            basin_id (str): 盆地编号，like "01013500"
            area_id (str): 区域编号，like "01"
            data_dir(str): 根目录
            seq_len (int): 输入序列长度 (encoder or decoder length)
            tar_len (int): 预测长度
            train_end...(str): 结束日期 like '2000-10-01'
            mode (str): 'train', 'val', or 'test' 用于划分数据
        """
        self.seq_len = seq_len
        self.target_len = tar_len

        'dataset/processed_data/01/01013500.csv'

        file_path = data_dir + f"/{area_id}/{basin_id}.csv"
        file_path_z_score = data_dir + "/all_attr.csv"

        df = pd.read_csv(file_path)
        df_z_score = pd.read_csv(file_path_z_score)

        # get static mean and std
        x_mean_static = df_z_score.iloc[:, 1:].mean().values
        x_std_static = df_z_score.iloc[:, 1:].std().values

        # split by time
        train_df = df.loc[(config['val_end'] < df['Date']) & (df['Date'] <= config['train_end'])].iloc[:, 1:].copy()
        val_df = df.loc[(config['test_end'] < df['Date']) & (df['Date'] <= config['val_end'])].iloc[:, 1:].copy()
        test_df = df.loc[(config['begin'] <= df['Date']) & (df['Date'] <= config['test_end'])].iloc[:, 1:].copy()

        # get the mean and std of train set
        train_df_y = train_df['flow(mm/day)'].values
        train_df_x = train_df.drop(columns=['flow(mm/day)']).values

        train_df_x_meteo = train_df_x[:, :5]
        train_df_x_static = train_df_x[:, 5:]

        x_mean_meteo = np.nanmean(train_df_x_meteo, axis=0, keepdims=True)
        x_std_meteo = np.nanstd(train_df_x_meteo, axis=0, keepdims=True)

        # self.x_mean = np.nanmean(train_df_x, axis=0, keepdims=True)
        # self.x_std = np.nanstd(train_df_x, axis=0, keepdims=True)
        self.y_mean = np.nanmean(train_df_y, axis=0, keepdims=True)
        self.y_std = np.nanstd(train_df_y, axis=0, keepdims=True)

        # get df_x and df_y (encoder input and decoder input)
        if mode == 'train':
            df = train_df
        elif mode == 'val':
            df = val_df
        else:
            df = test_df
        y_data = df['flow(mm/day)'].values.reshape(-1, 1)

        x_data = df.drop(columns=['flow(mm/day)']).values
        x_data_meteo = x_data[:, :5]
        x_data_static = x_data[:, 5:]

        # z-score
        x_data_x_meteo = (x_data_meteo - x_mean_meteo) / (x_std_meteo + eps)
        x_data_static = (x_data_static - x_mean_static) / (x_std_static + eps)
        y_data = (y_data - self.y_mean) / (self.y_std + eps)

        self.y_data = y_data
        self.x_data = np.concatenate((x_data_x_meteo, x_data_static), axis=1)

        self.length = len(self.x_data) - self.seq_len + 1

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

    def __len__(self):
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

    file_path_z_score = config['path'] + "/all_attr.csv"
    df_z_score = pd.read_csv(file_path_z_score)
    x_mean_static = df_z_score.iloc[:, 1:].mean().values
    x_std_static = df_z_score.iloc[:, 1:].std().values

    dataset = SingleBasinDataset(
        basin_id=basin_id,
        area_id=area_id,
        data_dir=config['path'],
        seq_len=config['seq_len'],
        tar_len=config['tar_len'],
        config=config['split'],
        mode=mode,
        x_mean_static=x_mean_static,
        x_std_static=x_std_static
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(mode == 'train'),  # 训练时打乱，验证测试时不打乱
        num_workers=config['num_workers']
    )

    return loader

def get_global_dataloader(config, mode='train'):
    all_basin = read_all_basin(config)
    dataset_list = []

    file_path_z_score = config['path'] + "/all_attr.csv"
    df_z_score = pd.read_csv(file_path_z_score)
    x_mean_static = df_z_score.iloc[:, 1:].mean().values
    x_std_static = df_z_score.iloc[:, 1:].std().values

    for area_id, basin_id in all_basin:
        try:
            ds = SingleBasinDataset(
                basin_id=basin_id,
                area_id=area_id,
                data_dir=config['path'],
                seq_len=config['seq_len'],
                tar_len=config['tar_len'],
                config=config['split'],
                mode=mode,
                x_mean_static=x_mean_static,
                x_std_static=x_std_static
            )
            if len(ds) > 0:
                dataset_list.append(ds)
        except Exception as e:
            continue

    global_dataset = ConcatDataset(dataset_list)

    loader = DataLoader(
        global_dataset,
        batch_size=config['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=config['num_workers'],
        pin_memory=True
    )
    return loader

'''
import yaml
config = yaml.load(open('./config/config.yaml', 'r'), Loader=yaml.FullLoader)
dt = get_dataloader(config['data'], '01013500', '01', 'train')

for x,y,z in dt:
    print(x[0][0])
    break
'''