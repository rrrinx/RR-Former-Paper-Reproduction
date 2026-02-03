## 简要介绍
本项目是对于论文[《RR-Former: Rainfall-runoff modeling based on Transformer》](https://www.sciencedirect.com/science/article/abs/pii/S0022169422003560)的简单复现

原作者：Hanlin Yin, Zilong Guo, Xiuwei Zhang, Jiaojiao Chen, Yanning Zhang

由于本人经验不足，项目结构较为繁琐，其中的`exp1`和`exp2`对应论文中的`individual model`和`regional model`。而对于`exp1`，`exp1`，
`exp1 pretrain`，`exp1 pretrain & fine-tune`则对应论文中`individual model`的
`RR-Former without pretrain`，`RR-Former without fine-tune`和`RR-Former`三种模型。如果有需要，之后再来优化项目结构

由于对于上传数据大小的限制，本项目不包含训练结果的pt文件还有原始的数据文件

## 运行
在各个实验文件夹下打开终端运行:
```bash
    python main.py
```
以进行训练

另外在`exp1`和`exp1 pretrain & fine-tune`中对不同的盆地进行了不同段的划分，并使用多进程以加速模型训练

可以打开多个终端，并输入
```bash
    python main.py --seg 1
```
运行

seg之后输入参数以训练不同段的盆地模型

其中`exp1`总共分为了8段，`exp1 pretrain & fine-tune`总共分为了4段

## 结果
当前版本仅仅计算了`nse`作为评价依据，如果有必要之后会补上其他评价标准
### exp1
在此仅仅贴上`pretrain & fine-tune`模型与原论文对比的结果

其他数据都可以在res/exp1/res_table.csv中找到

|      evaluation Metric      | 1st-day-ahead | 2nd-day-ahead | 3rd-day-ahead | 4th-day-ahead | 5th-day-ahead | 6th-day-ahead | 7th-day-ahead |
|:---------------------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| median of nse (paper's res) |    0.8265     |    0.7832     |    0.7674     |    0.7582     |    0.7484     |    0.7423     |    0.7282     |
|   median of nse (my res)    |    0.8213     |    0.7794     |    0.7652     |    0.7627     |    0.7550     |    0.7489     |    0.7217     |
|  mean of nse(paper's res)   |    0.7904     |    0.7421     |    0.7241     |    0.7107     |    0.7003     |    0.6901     |    0.6734     |
|    mean of nse(my res)      |    0.7897     |    0.7545     |    0.7374     |    0.7262     |    0.7171     |    0.7082     |    0.6879     |


### exp2
|      evaluation Metric      | 1st-day-ahead | 2nd-day-ahead | 3rd-day-ahead | 4th-day-ahead | 5th-day-ahead | 6th-day-ahead | 7th-day-ahead |
|:---------------------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| median of nse (paper's res) |     0.805     |     0.760     |     0.746     |     0.739     |     0.730     |     0.725     |     0.708     |
|   median of nse (my res)    |     0.792     |     0.756     |     0.742     |     0.737     |     0.733     |     0.728     |     0.709     |
|  mean of nse(paper's res)   |     0.773     |     0.729     |     0.713     |     0.702     |     0.692     |     0.685     |     0.674     |
|    mean of nse(my res)      |     0.759     |     0.725     |     0.712     |     0.702     |     0.696     |     0.691     |     0.676     |

## 讨论
首先本实验存在较多不严谨的地方

在exp1中考虑到存在缺失的问题，本项目没有使用论文中按照固定日期划分训练集，测试集和验证集的划分方法，而是按照比例划分。划分的方式不同导致模型结果的对比意义不大

另外在转换`runoff`的单位时(从cfs到mm/day)，所要用到的camels数据集中关于盆地的面积有两个量：`area gage2`以及`area geospa fabric`。论文中没有提及这一点该选用哪个。
本项目在exp1中选择了后者，在exp2中选择前者。但是由于重新跑一遍的时间代价不小，暂时就先保留目前有问题的`runoff`。当然这个问题进一步使得本复现的意义缩减。

考虑复现结果，本项目的复现结果相较论文，nse指标一般稍微低一点，但都在0.02以内，大部分在0.01以内，少部分甚至优于论文结果。
由于论文并没有详细介绍其训练的策略和参数设置，如epoch数，学习率调度器等等，本人只能按照训练该模型的经验做出调整，目前拿出的便是自己暂时得到的最优模型。

最后值得一提的是，exp2中的448个盆地的列表自己实在是找不到，所以直接接用了[论文作者的github仓库](https://github.com/iThronne/RR-Former)中的文件448basins_list.txt

## 可以考虑的改进方向
首先是修改exp1的划分方式。确认好应该选用哪一个盆地面积后统一exp1和exp2的面积换算，重新训练模型。

其次是优化训练方式，优化学习率调度器，进一步提高模型能力。

除此之外还可以选择多个种子进行实验，最后结果取平均，以此更加全面的评价模型能力。

还需要加上论文中提及的其他评价指标，`RESM`、`ATPE-2%`、`Blas`，并绘制相关图像，将复现结果可视化。

