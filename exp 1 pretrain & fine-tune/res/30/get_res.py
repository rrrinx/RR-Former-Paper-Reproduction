import pandas as pd

df1 = pd.read_csv('res_1.csv')
df2 = pd.read_csv('res_2.csv')
df3 = pd.read_csv('res_3.csv')
df4 = pd.read_csv('res_4.csv')

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

for col in df.columns[1:]:
    print(df[col].mean())


print('\n')

for col in df.columns[1:]:
    print(df[col].median())

