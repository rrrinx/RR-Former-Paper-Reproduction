import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from myloss import NSELoss
from utils import calc_nse_gpu
from tqdm import tqdm


class Trainer:
    def __init__(self, model,
                 train_loader, val_loader, config,
                 basin_id=None, area_id=None,
                 ):
        self.device = config['device']
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 1. 定义损失函数
        self.criterion = NSELoss()

        # 2. 定义优化器
        self.lr = float(config['learning_rate'])
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay']),
        )

        self.epochs = config['epochs']
        self.warmup_epochs = 10

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=1e-6
        )

        self.save_path = config['save_dir'] + '/best_model.pth'

        folder_path = os.path.dirname(self.save_path)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        self.best_val_loss = float('inf')

    def train_one_epoch(self, epoch_idx):
        """训练一个 Epoch"""
        self.model.train()  # 开启训练模式 (启用 Dropout)

        if epoch_idx <= self.warmup_epochs:
            # --- Warmup 阶段 ---
            # 学习率从 0 线性增加到 target_lr
            lr_scale = (epoch_idx + 1) / self.warmup_epochs
            current_lr = self.lr * lr_scale

            # 手动更新优化器中的学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            # --- Cosine Decay 阶段 ---
            # 在 warmup 结束后，交给 PyTorch 的 scheduler 处理
            # 注意：CosineAnnealingLR 需要在 step() 时更新，我们在本函数末尾调用
            pass
        
            # 打印当前学习率 (方便你监控)
        # current_lr = self.optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch_idx+1} LR: {current_lr:.6f}")

        total_loss = 0

        for x_batch, y_batch, y_batch_valid in self.train_loader:
            # 1. 数据搬家到 GPU
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_batch_valid = y_batch_valid.to(self.device)

            input_ecd = x_batch
            input_dcd = y_batch

            # 清空梯度
            self.optimizer.zero_grad()

            # 前向传播
            pred = self.model(input_ecd, input_dcd)

            # 计算 Loss
            loss = self.criterion(pred, y_batch_valid)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('clip_grad', 1.0)
            )

            #更新参数
            self.optimizer.step()

            total_loss += loss.item()

        if epoch_idx > self.warmup_epochs:
            self.scheduler.step()
        return total_loss / len(self.train_loader)

    def validate(self):
        """验证过程 (不计算梯度)"""
        self.model.eval()
        pred_list = []
        val_list = []

        data = self.val_loader

        with torch.no_grad():
            for x_batch, y_batch, y_batch_valid in data:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_batch_valid = y_batch_valid.to(self.device)
                # print(y_batch.shape)

                pred = self.model(x_batch, y_batch)
                # if y_batch.isnan().sum() > 0:
                #     self.model(x_batch, y_batch)
                #     break
                # print(pred.isnan().sum())
                pred_list.append(pred)  # pred_list.append(pred.cpu().numpy())
                val_list.append(y_batch_valid)  # val_list.append(y_batch_valid.cpu().numpy())

        all_pred = torch.cat(pred_list, dim=0)

        all_val = torch.cat(val_list, dim=0)

        nse_score = calc_nse_gpu(all_pred, all_val)  # nse_score = calc_nse(all_pred, all_val)

        return nse_score

    def fit(self):
        """主训练循环：跑 epochs 轮"""
        epochs = self.config['epochs']
        patience = self.config.get('patience', 50)
        patience_counter = 0

        for epoch in range(1, epochs + 1):



            # 1. 训练
            train_loss = self.train_one_epoch(epoch)

            # 2. 验证
            val_loss = self.validate()
            # test_loss = self.test()


            # 3. 打印进度
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

            # 4. 保存最佳模型 (Model Checkpoint)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                torch.save(self.model.state_dict(), self.save_path)
                print(f"发现新低 Loss，模型已保存至 {self.save_path}")
                patience_counter = 0  # 重置早停计数器
            else:
                patience_counter += 1

            # 5. 早停机制
            if patience_counter >= patience:
                print(f"验证集 Loss 连续 {patience} 轮未下降，触发早停。")
                break

    def test(self, test_loader):
        """验证过程 (不计算梯度)"""
        self.model.eval()
        days = self.config['days']
        pred_list = []
        val_list = []

        data = test_loader

        with torch.no_grad():
            for x_batch, y_batch, y_batch_valid in data:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_batch_valid = y_batch_valid.to(self.device)

                pred = self.model(x_batch, y_batch)

                pred_list.append(pred)  # pred_list.append(pred.cpu().numpy())
                val_list.append(y_batch_valid)  # val_list.append(y_batch_valid.cpu().numpy())

        all_pred = torch.cat(pred_list, dim=0)
        all_val = torch.cat(val_list, dim=0)
        std, mean = data.dataset.y_std, data.dataset.y_mean
        std = torch.tensor(std, device=self.device, dtype=torch.float32)  #
        mean = torch.tensor(mean, device=self.device, dtype=torch.float32)  #

        all_pred = all_pred * (std + 1e-5) + mean
        all_val = all_val * (std + 1e-5) + mean

        nse = calc_nse_gpu(all_pred, all_val)
        nse_list_days = []

        for i in range(days):
            day_pred = all_pred[:, i, :]
            day_val = all_val[:, i, :]

            day_nse = calc_nse_gpu(day_pred, day_val)
            nse_list_days.append(day_nse)

        return nse, nse_list_days



