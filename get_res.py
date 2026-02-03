import pandas as pd
import os
from plot import plot_cdf_matplotlib

""" exp 1 """
# rr_a -> without pretrain rr_b -> without finetune rr_c -> pretrain & finetune
PATH_A = './exp1/res/res.csv'
PATH_B = 'exp1 pretrain/res/res.csv'
PATH_C = 'exp 1 pretrain & fine-tune/res/50/res.csv'

PATH_OUT = './res/exp1'
os.makedirs(PATH_OUT, exist_ok=True)

df_rr_a = pd.read_csv(PATH_A)
df_rr_b = pd.read_csv(PATH_B)
df_rr_c = pd.read_csv(PATH_C)
df_rr_a.iloc[:, 1:] = 1 - df_rr_a.iloc[:, 1:]
df_rr_b.iloc[:, 1:] = 1 - df_rr_b.iloc[:, 1:]
df_rr_c.iloc[:, 1:] = 1 - df_rr_c.iloc[:, 1:]


print(df_rr_a.head())

df_rr_median = []
df_rr_mean = []

for model in [df_rr_a, df_rr_b, df_rr_c]:
    df_rr_median.append(model[1:].median())
    df_rr_mean.append(model[1:].mean())
model_list = ['without pretrain', 'without fine-tune', 'pretrain & fine-tune']
df_median = pd.DataFrame(df_rr_median)
df_mean = pd.DataFrame(df_rr_mean)
model_name = pd.DataFrame(model_list, columns=['model'])
df_median.insert(0, 'model', model_name)
df_mean.insert(0, 'model', model_name)



df_list = [df_median.set_index('model'), df_mean.set_index('model')]
df_combined = pd.concat(df_list, keys=['median', 'mean'], axis=0)

df_combined.to_csv(os.path.join(PATH_OUT, 'res_table.csv'))

plot_cdf_matplotlib([df_rr_a, df_rr_b, df_rr_c], df_rr_a.columns[1:], model_list, PATH_OUT+'/cdf.png')


""" exp 2 """
PATH = './exp2/res/res.csv'
PATH_OUT_EXP2 = './res/exp2'
os.makedirs(PATH_OUT_EXP2, exist_ok=True)

df_rr = pd.read_csv(PATH)
df_rr.iloc[:, 1:] = 1 - df_rr.iloc[:, 1:]

df_rr_mean_exp2 = df_rr[1:].mean()
df_rr_median_exp2 = df_rr[1:].median()

df_list_exp2 = [df_rr_median_exp2,df_rr_mean_exp2]
df_combined_exp2 = pd.concat(df_list_exp2, keys=['median', 'mean'], axis=0)

df_combined_exp2.to_csv(os.path.join(PATH_OUT_EXP2, 'res_table.csv'))

