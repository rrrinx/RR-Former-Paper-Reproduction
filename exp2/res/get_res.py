import pandas as pd

df = pd.read_csv('res.csv')

for columns in df.columns[1:]:
    print(df[columns].mean())

print()

for columns in df.columns[1:]:
    print(df[columns].median())

