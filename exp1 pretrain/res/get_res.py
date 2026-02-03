import pandas as pd

df = pd.read_csv('res.csv')

for col in df.columns[1:]:
    print(df[col].mean())


print('\n')

for col in df.columns[1:]:
    print(df[col].argmax())