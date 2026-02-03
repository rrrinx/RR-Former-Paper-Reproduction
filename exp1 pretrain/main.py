import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataloader import get_global_dataloader, read_all_basin, get_dataloader
import yaml
import torch
from trainer import Trainer
from model import RR_Former
from tqdm import tqdm
import time
import csv

def main():
    config = yaml.load(open('./config/config.yaml', 'r'), Loader=yaml.FullLoader)

    start_time = time.time()
    all_basin = read_all_basin(config['data'])
    basin_res = []
    for area_id, basin_id in all_basin:
        basin_res.append({"basin": f'{area_id}/{basin_id}'})

    torch.manual_seed(config['seed'])
    train_loader = get_global_dataloader(config['data'], mode='train')
    val_loader = get_global_dataloader(config['data'], mode='val')
    model = RR_Former(config['model'])
    # print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    trainer = Trainer(model, train_loader, val_loader, config['train'])
    # loss = trainer.validate_test(True)
    # print(loss)

    trainer.fit()

    # plot_results(train_loss_list[10:], val_loss_list[10:],
    # label_x='train_loss', label_y='val_loss', name='loss.png')

    state_dict = torch.load(config['train']['save_dir'] + f'/best_model.pth')
    model.load_state_dict(state_dict)
    for idx, (area_id, basin_id) in enumerate(all_basin):
        test_loader = get_dataloader(config['data'], mode='test', area_id=area_id, basin_id=basin_id)
        nse, nse_list_day = trainer.test(test_loader)
        basin_res[idx]['nse'] = nse
        for i in range(config['days']):
            basin_res[idx][f'nse_{i + 1}day(s)'] = nse_list_day[i]

    end_time = time.time()
    print('Total time: {:.2f}s'.format(end_time - start_time))
    res_path = config['res_path']
    os.makedirs(res_path, exist_ok=True)

    with open(res_path + f'/res.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["basin"] +
                                              [f'nse_{i + 1}day(s)' for i in range(config['days'])] + ['nse_all'])
        writer.writeheader()
        writer.writerows(basin_res)

if __name__ == '__main__':
    main()

