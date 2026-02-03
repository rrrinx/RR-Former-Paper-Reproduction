import os
import pandas as pd
import numpy as np
folder_a = '.\\maurer'
folder_b = '.\\usgs_streamflow'
output_folder = '.\\processed_data'

flag = 0

clim = pd.read_csv('camels_clim.txt', sep=';', dtype=str)
geol = pd.read_csv('camels_geol.txt', sep=';', dtype=str)
soil = pd.read_csv('camels_soil.txt', sep=';', dtype=str)
topo = pd.read_csv('camels_topo.txt', sep=';', dtype=str)
vege = pd.read_csv('camels_vege.txt', sep=';', dtype=str)
with open('.\\448basins_list.txt', 'r', encoding='utf-8') as fst:
    set0 = set(line.strip() for line in fst if line.strip())

lack_flow = []
lack_clim = []
lack_geol = []
lack_soil = []
lack_topo = []
lack_vege = []

all_basin = []


num = 0
c = 86400 * 28.317 / 1e6
# preprocess the attr
target_line_idx = 3  # begin with 0
new_content = "Year Mnth Day Hr	Dayl(s)	PRCP(mm/day)	SRAD(W/m2)	SWE(mm)	Tmax(C)	Tmin(C)	Vp(Pa)"

# 遍历文件夹
for root, dirs, files in os.walk(folder_a):
    for file in files:
        if file.endswith(".txt"):
            path_a = os.path.join(root, file)

            with open(path_a, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) > target_line_idx:
                lines[target_line_idx] = new_content + '\n'
                with open(path_a, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

list_maurer = []
list_usgs_streamflow = []
for root, dirs, files in os.walk(folder_a):
    for file in files:
        if file.endswith(".txt"):
            path_a = os.path.join(root, file)
            rel_path = os.path.relpath(path_a, folder_a)
            area_id = rel_path[:rel_path.find('\\')]
            basin_id = rel_path[rel_path.rfind('\\')+1:rel_path.find('_')]
            list_maurer.append(basin_id)

for root, dirs, files in os.walk(folder_b):
    for file in files:
        if file.endswith(".txt"):
            path_b = os.path.join(root, file)
            rel_path = os.path.relpath(path_b, folder_b)
            area_id = rel_path[:rel_path.find('\\')]
            basin_id = rel_path[rel_path.rfind('\\')+1:rel_path.find('_')]
            list_usgs_streamflow.append(basin_id)

set1 = set(list_maurer)
set2 = set(list_usgs_streamflow)


print(len(set0 & set1))
print(len(set0 & set2))

all_attr = []


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

            df_a = pd.read_csv(path_a, sep='\s+', skiprows=3)
            df_b = pd.read_csv(path_b, sep='\s+', header=None)

            df_a.rename(columns={
                'PRCP(mm/day)': 'prcp(mm/day)',
                'SRAD(W/m2)': 'srad(W/m2)',
                'Tmax(C)': 'tmax(C)',
                'Tmin(C)': 'tmin(C)',
                'Vp(Pa)': 'vp(Pa)',
                'Year': 'year',
                'Mnth': 'month',
                'Day': 'day',
            }, inplace=True)
            df_a['Date'] = pd.to_datetime(df_a[['year', 'month', 'day']])
            df_a.set_index('Date', inplace=True)
            df_a.drop(columns=['year', 'month', 'day', 'Hr', 'SWE(mm)', 'Dayl(s)'], inplace=True)
            df_b.rename(columns={
                1: 'year',
                2: 'month',
                3: 'day'
            }, inplace=True)

            df_b['Date'] = pd.to_datetime(df_b[['year', 'month', 'day']])
            df_b.set_index('Date', inplace=True)
            # df_b.drop(df_b[df_b[5] == 'M'].index, inplace=True)
            # df_b.loc[:, 4] = df_b.loc[:, 4] * c / basin_area
            df_b.loc[df_b[5] == 'M', 4] = np.nan

            df_b_final = df_b[4].copy()
            df_b_final.rename('flow(mm/day)', inplace=True)
            df_merged = pd.concat([df_a, df_b_final], axis=1, join='inner')

            last_index = df_merged['flow(mm/day)'].last_valid_index()
            df_merged = df_merged.loc[:last_index]


            attr_dict = {'basin_id' : basin_id}
            # camels_topo.txt
            if topo[topo['gauge_id'] == basin_id].shape[0] == 0:
                lack_topo.append(basin_id)
                continue

            df_merged.loc[:, 'elev_mean'] = topo[topo['gauge_id'] == basin_id].elev_mean.item()
            attr_dict['elev_mean'] = topo[topo['gauge_id'] == basin_id].elev_mean.item()

            df_merged.loc[:, 'slope_mean'] = topo[topo['gauge_id'] == basin_id].slope_mean.item()
            attr_dict['slope_mean'] = topo[topo['gauge_id'] == basin_id].slope_mean.item()

            area = float(topo[topo['gauge_id'] == basin_id].area_gages2.item())
            df_merged.loc[:, 'area'] = area
            attr_dict['area'] = area

            df_merged.loc[:, 'flow(mm/day)'] = df_merged.loc[:, 'flow(mm/day)'] * c / area



            
            # camels_clim.txt
            if clim[clim['gauge_id'] == basin_id].shape[0] == 0:
                lack_clim.append(basin_id)
                continue

            for attr in ['p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow',
                         'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur']:
                df_merged.loc[:, attr] = clim[clim['gauge_id'] == basin_id][attr].item()
                attr_dict[attr] = clim[clim['gauge_id'] == basin_id][attr].item()


            # camels_geol.txt
            if geol[geol['gauge_id'] == basin_id].shape[0] == 0:
                lack_geol.append(basin_id)
                continue

            for attr in ['carbonate_rocks_frac', 'geol_permeability']:
                df_merged.loc[:, attr] = geol[geol['gauge_id'] == basin_id][attr].item()
                attr_dict[attr] = geol[geol['gauge_id'] == basin_id][attr].item()

            # camels_soil.txt
            if soil[soil['gauge_id'] == basin_id].shape[0] == 0:
                lack_soil.append(basin_id)
                continue

            for attr in ['soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
                         'soil_conductivity', 'max_water_content', 'sand_frac',
                         'silt_frac', 'clay_frac']:
                df_merged.loc[:, attr] = soil[soil['gauge_id'] == basin_id][attr].item()
                attr_dict[attr] = soil[soil['gauge_id'] == basin_id][attr].item()


            # camels_vege.txt
            if vege[vege['gauge_id'] == basin_id].shape[0] == 0:
                lack_vege.append(basin_id)
                continue

            for attr in ['frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff']:
                df_merged.loc[:, attr] = vege[vege['gauge_id'] == basin_id][attr].item()
                attr_dict[attr] = vege[vege['gauge_id'] == basin_id][attr].item()



            if len(df_merged) < 0.1 * len(df_a) or len(df_merged) < 0.1 * len(df_b):
                continue
            

            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            df_merged.to_csv(path_out, index=True, header=True)
            if basin_id in set0:
                all_basin.append((area_id, basin_id))
                num += 1
                all_attr.append(attr_dict)

print(num)

data_to_save = {
    'all_basin': all_basin,
    'lack_flow': lack_flow,
    'lack_clim': lack_clim,
    'lack_geol': lack_geol,
    'lack_soil': lack_soil,
    'lack_topo': lack_topo,
    'lack_vege': lack_vege,
}

for k, v in data_to_save.items():
    with open(f'./processed_data/{k}.txt', 'w', encoding='utf-8') as f:
        for item in v:
            f.write(str(item) + '\n')

df = pd.DataFrame(all_attr)
print(df.shape)
print(df.head())
z_score = {}
mean = []
std = []
for columns in df.columns[1:]:
    mean.append(df[columns].mean())
for columns in df.columns[1:]:
    std.append(df[columns].std())


df.to_csv(f'./processed_data/all_attr.csv', index=False)
