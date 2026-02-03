import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataloader import get_dataloader, read_all_basin
import yaml
import torch
from trainer import Trainer
from model import RR_Former
from tqdm import tqdm
import csv
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg', type=int, default=1, help='Segment id: 1, 2, 3, or 4')
    args = parser.parse_args()
    config = yaml.load(open('./config/config.yaml', 'r'), Loader=yaml.FullLoader)

    if type(config['execution_target']) is str and config['execution_target'] == 'all':
        basin_list = read_all_basin(config['data'], args.seg)
    else:
        basin_list = config['execution_target']
        basin_list = [tuple(item) for item in basin_list]

    basin_res = []
    for area_id, basin_id in basin_list:
        basin_res.append({"basin": f'{area_id}/{basin_id}'})

    start_time = time.time()
    pbar = enumerate(tqdm(basin_list, desc="Training", leave=False))
    avg_loss = 0.0
    for idx, (area_id, basin_id) in pbar:
        torch.manual_seed(config['seed'])
        train_loader = get_dataloader(config['data'], area_id=area_id, basin_id=basin_id, mode='train')
        val_loader = get_dataloader(config['data'], area_id=area_id, basin_id=basin_id, mode='val')
        test_loader = get_dataloader(config['data'], area_id=area_id, basin_id=basin_id, mode='test')
        model = RR_Former(config['model'])
        # print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
        trainer = Trainer(model, train_loader, val_loader, test_loader, basin_id, area_id, config['train'])
        # loss = trainer.validate_test(True)
        # print(loss)

        trainer.fit()

        # plot_results(train_loss_list[10:], val_loss_list[10:],
        # label_x='train_loss', label_y='val_loss', name='loss.png')

        state_dict = torch.load(config['train']['save_dir'] + f'/{area_id}/{basin_id}_best_model.pth')
        model.load_state_dict(state_dict)

        nse, nse_list_day = trainer.test()
        avg_loss += nse
        basin_res[idx]['nse_all'] = nse
        #plot_test(pred_list, test_list, day=day, basin_id=basin_id, area_id=area_id, config=config['img'])
        for i in range(config['days']):
            basin_res[idx][f'nse_{i + 1}day(s)'] = nse_list_day[i]

    end_time = time.time()
    print('Total time: {:.2f}s'.format(end_time - start_time))
    # print('Avg loss: {:.4f}'.format(avg_loss / len(basin_list)))
    res_path = config['res_path']
    os.makedirs(res_path, exist_ok=True)

    with open(res_path + f'/res_{args.seg}.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["basin"] +
                                              [f'nse_{i + 1}day(s)' for i in range(config['days'])] + ['nse_all'])
        writer.writeheader()
        writer.writerows(basin_res)

if __name__ == '__main__':
    main()

