import matplotlib.pyplot as plt
import numpy as np
def plot_cdf_matplotlib(df_list, col_list, model_list, save_path):
    fig, ax_list = plt.subplots(4, 2, figsize=(10, 10))
    ax_list = ax_list.flatten()

    colors = plt.cm.Set1(np.linspace(0, 1, len(df_list)))

    for i, col in enumerate(col_list):
        ax = ax_list[i]
        for j, df in enumerate(df_list):
            # 获取数据并排序
            data_sorted = np.sort(df[col].dropna())

            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            # 绘制CDF曲线
            ax.plot(data_sorted, cdf,
                     label=model_list[j],
                     color=colors[j],
                     linewidth=2)
            # ax.legend(title='model', loc='best')
            ax.set_title(str(i+1) + '-Day-Ahead' if i != 7 else 'ALL')
            ax.grid(True, alpha=0.3)

    fig.legend(title='model',
               labels=model_list,
               loc='upper center',
               bbox_to_anchor=(0.5, 0),
               ncol=len(model_list),
               fontsize=10)
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 为底部图例留出空间


    plt.legend(title='model', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
