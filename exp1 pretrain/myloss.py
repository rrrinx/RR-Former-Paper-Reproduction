import torch
import torch.nn as nn

class NSELoss(nn.Module):

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        mask = ~torch.isnan(y_true)
        # 为了保持维度以便后续广播计算，我们暂时把 NaN 填为 0
        y_true_filled = torch.nan_to_num(y_true, nan=0.0)
        y_pred_filled = y_pred * mask.float()

        squared_error = torch.sum((y_pred_filled - y_true_filled) ** 2, dim=1)
        valid_counts = mask.sum(dim=1)
        valid_counts = torch.clamp(valid_counts, min=1.0)
        sample_means = torch.sum(y_true_filled, dim=1) / valid_counts
        deviations = (y_true_filled - sample_means.unsqueeze(1)) * mask.float()
        squared_deviations = torch.sum(deviations ** 2, dim=1)

        per_sample_loss = squared_error / (squared_deviations + self.epsilon)

        valid_samples_mask = (mask.sum(dim=1) > 0).float()
        total_loss = torch.sum(per_sample_loss * valid_samples_mask)
        num_valid_samples = torch.sum(valid_samples_mask)

        if num_valid_samples == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        return total_loss / num_valid_samples

