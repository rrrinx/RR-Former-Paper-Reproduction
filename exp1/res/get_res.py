import pandas as pd

df1 = pd.read_csv('res_1.csv')
df2 = pd.read_csv('res_2.csv')
df3 = pd.read_csv('res_3.csv')
df4 = pd.read_csv('res_4.csv')
df5 = pd.read_csv('res_5.csv')
df6 = pd.read_csv('res_6.csv')
df7 = pd.read_csv('res_7.csv')
df8 = pd.read_csv('res_8.csv')

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

for col in df.columns[1:]:
    print(df[col].mean())


print('\n')

for col in df.columns[1:]:
    print(df[col].median())

df.to_csv('res.csv', index=False)


