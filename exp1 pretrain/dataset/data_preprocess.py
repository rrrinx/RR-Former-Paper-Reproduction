import os
import pandas as pd
import numpy as np
folder_a = '.\daymet'
folder_b = '.\\usgs_streamflow'
output_folder = '.\processed_data'

area = pd.read_csv('camels_topo.txt', sep=';', dtype=str)
flag = 0
nums = 0

c = 86400 * 28.317 / 1e6

lack_flow = []
lack_area = []
all_basin = []

for root, dirs, files in os.walk(folder_a):
    for file in files:
        if file.endswith(".txt"):
            flag = 1
            path_a = os.path.join(root, file)
            rel_path = os.path.relpath(path_a, folder_a)
            # print(rel_path)

            path_out = os.path.join(output_folder, rel_path[:rel_path.find('_')])
            path_out = os.path.splitext(path_out)[0] + '.csv'
            area_id = rel_path[:rel_path.find('\\')]
            basin_id = rel_path[rel_path.rfind('\\')+1:rel_path.find('_')]
            # print(area_id, basin_id)

            rel_path = rel_path[:rel_path.find('_')] + '_streamflow_qc.txt'
            path_b = os.path.join(folder_b, rel_path)




            if not os.path.exists(path_b):
                lack_flow.append(basin_id)
                continue
            if area[area['gauge_id'] == basin_id].shape[0] == 0:
                lack_area.append(basin_id)
                continue



            df_a = pd.read_csv(path_a, sep='\s+', skiprows=3)
            df_b = pd.read_csv(path_b, sep='\s+', header=None)

            basin_area = float(area[area['gauge_id'] == basin_id].area_geospa_fabric.item())

            df_a.rename(columns={
                'Year': 'year',
                'Mnth': 'month',
                'Day': 'day',
            }, inplace=True)
            df_a['Date'] = pd.to_datetime(df_a[['year', 'month', 'day']])
            df_a.set_index('Date', inplace=True)
            df_a.drop(columns=['year', 'month', 'day', 'Hr', 'swe(mm)', 'dayl(s)'], inplace=True)
            df_b.rename(columns={
                1: 'year',
                2: 'month',
                3: 'day'
            }, inplace=True)

            df_b['Date'] = pd.to_datetime(df_b[['year', 'month', 'day']])
            df_b.set_index('Date', inplace=True)
            # df_b.drop(df_b[df_b[5] == 'M'].index, inplace=True)
            df_b.loc[:, 4] = df_b.loc[:, 4] * c / basin_area
            df_b.loc[df_b[5] == 'M', 4] = np.nan



            df_b_final = df_b[4].copy()
            df_b_final.rename('flow(mm/day)', inplace=True)
            df_merged = pd.concat([df_a, df_b_final], axis=1, join='inner')

            last_index = df_merged['flow(mm/day)'].last_valid_index()
            df_merged = df_merged.loc[:last_index]



            if df_merged.index.to_series().diff().value_counts().nunique() != 1:
                print(f"{rel_path} 不连续")

            if len(df_merged) < 0.1 * len(df_a):
                continue

            if len(df_merged) != len(df_a) or len(df_merged) != len(df_b):
                print(f"[警告] 行数不匹配: {rel_path} (A:{len(df_a)}, B:{len(df_b)}, merged:{len(df_merged)})")

            # 确保目录存在
            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            df_merged.to_csv(path_out, index=True, header=True)
            all_basin.append((area_id,basin_id))
            nums += 1

print(nums)

with open('./processed_data/lack_flow.txt', 'w', encoding='utf-8') as f:
    for item in lack_flow:
        f.write(str(item) + '\n')

with open('./processed_data/lack_area.txt', 'w', encoding='utf-8') as f:
    for item in lack_area:
        f.write(str(item) + '\n')

with open('./processed_data/all_basin.txt', 'w', encoding='utf-8') as f:
    for item in all_basin:
        f.write(str(item) + '\n')
