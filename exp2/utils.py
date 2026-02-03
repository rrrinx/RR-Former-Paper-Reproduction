import numpy as np
import torch

def calc_nse(pred, val, eps=0.1):
    """
    计算 Nash-Sutcliffe Efficiency (NSE)
    :param preds: 预测值数组 (N, ...)
    :param val: 真实值数组 (N, ...)
    :return: NSE score
    """
    # 1. 确保形状一致并展平，方便统一计算
    # 这里的 reshape(-1) 意味着把所有时间步、所有样本都拉成一条直线来对比
    # 如果你想针对每个时间步分别算，就不展平
    pred = pred.reshape(-1)
    val = val.reshape(-1)

    # 2. 处理可能存在的 NaN (水文数据中常见)
    # 如果 targets 里有 NaN，计算结果会变成 NaN，这里做一个简单的掩码处理
    val_mask = ~np.isnan(val)
    pred = pred[val_mask]
    val = val[val_mask]

    # 3. 计算分子分母
    numerator = np.sum((val - pred) ** 2)

    val_mean = np.mean(val)
    denominator = np.sum((val - val_mean) ** 2)

    return numerator / (denominator + eps)

def calc_rmse(pred, val, eps=1e-5):
    pred = pred.reshape(-1)
    val = val.reshape(-1)

    val_mask = ~np.isnan(val)
    pred = pred[val_mask]
    val = val[val_mask]

    numerator = np.sum((val - pred) ** 2)
    denominator = val.shape[0]

    return np.sqrt(numerator / (denominator + eps))

@torch.no_grad()
def calc_nse_gpu(pred, obs):
    """
    计算标准 NSE 指标 (Global Calculation)
    pred: [Total_Samples, T, 1]  (已拼接好的全量验证集预测)
    obs:  [Total_Samples, T, 1]  (已拼接好的全量验证集观测)
    """
    # 1. 展平所有数据，视为一个长序列
    pred_flat = pred.view(-1)
    obs_flat = obs.view(-1)
    # print("pred:1 ", pred_flat.shape)
    # print("obs:1 ", obs_flat.shape)
    mask = ~torch.isnan(obs_flat)
    pred_flat = pred_flat[mask]
    obs_flat = obs_flat[mask]
    # print("pred:2 ", pred_flat.shape)
    # print("obs:2 ", obs_flat.shape)

    # 2. 计算分子：残差平方和 (Sum of Squared Errors)
    numerator = torch.sum((pred_flat - obs_flat) ** 2)

    # 3. 计算分母：总方差 (Total Sum of Squares)
    # 注意：这里是用 全局均值 (Global Mean)
    obs_mean = torch.mean(obs_flat)
    denominator = torch.sum((obs_flat - obs_mean) ** 2)

    # 4. 计算 NSE
    # 仅加极小 eps 防止分母绝对为 0 (例如观测全是常数)
    # print(numerator, denominator)
    nse = (numerator / (denominator + 1e-6))

    return nse.item()