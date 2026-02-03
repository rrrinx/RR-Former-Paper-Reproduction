import matplotlib.pyplot as plt
import os
import numpy as np


def plot_loss(line_x, line_y, label_x, label_y, name):
    # 设置绘图风格 (可选)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.plot(line_x, label=label_x, color='blue', linewidth=2)
    plt.plot(line_y, label=label_y, color='red', linewidth=2)

    # 保存图片
    plt.savefig(name, dpi=300)
    print("图像已保存")


def plot_test(pred, obs,
              basin_id, area_id,
              day,
              config):
    pred = np.array(pred).flatten()
    obs = np.array(obs).flatten()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    ax1.plot(obs, label='Observation',
             color='black', linewidth=2, linestyle='-', alpha=0.8, zorder=1)
    ax1.plot(pred, label='Prediction',
             color='red', linewidth=1.5, linestyle='--', alpha=0.7, zorder=2)

    ax1.set_title('pred cmp obs', fontsize=16)
    ax1.set_xlabel('time step', fontsize=12)
    ax1.set_ylabel('run off', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.5)

    pred_part = pred[day*300:day*600]
    obs_part = obs[day*300:day*600]
    residual = pred_part - obs_part

    ax2.plot(obs_part, label='Observation',
             color='black', linewidth=2, linestyle='-', alpha=0.8, zorder=1)
    ax2.plot(pred_part, label='Prediction',
             color='red', linewidth=1.5, linestyle='--', alpha=0.7, zorder=2)

    ax2.set_title('pred cmp obs(600 - 1200)', fontsize=16)
    ax2.set_xlabel('time step', fontsize=12)
    ax2.set_ylabel('run off', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # === 下子图：残差图 ===
    # 画一条 0 基准线
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    # 画残差曲线，可以用细线或者散点
    ax3.plot(residual, color='blue', linewidth=1, alpha=0.6, label='Residual (Pred - Obs)')
    # 也可以用填充图来强调偏差
    # ax2.fill_between(range(len(residuals)), residuals, 0, where=(residuals>0), color='red', alpha=0.3)
    # ax2.fill_between(range(len(residuals)), residuals, 0, where=(residuals<0), color='blue', alpha=0.3)

    ax3.set_ylabel('Residual Error', fontsize=12)
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.5)
    plt.subplots_adjust(hspace=0.05)

    plt.tight_layout()
    save_path = config['save_dir'] + f'/{day}/{area_id}/{basin_id}_pred_obs_cmp.png'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"图像已保存为 {save_path}")
