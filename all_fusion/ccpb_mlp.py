import torch
import torch.nn as nn

class CCPB_MLP(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=6):
        super(CCPB_MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MLPLoss(nn.Module):
    def __init__(self, lambda_val=1.5):
        super(MLPLoss, self).__init__()
        self.reg_loss = nn.MSELoss(reduction='mean') 
        self.conf_loss = nn.MSELoss(reduction='mean') 
        self.lambda_val = lambda_val
        
    def forward(self, pred, target):
        """
        pred: [N, 6] -> [cx, cy, w, h, cls, conf]
        target: [N, 6] -> [gt_cx, gt_cy, gt_w, gt_h, gt_cls, gt_conf]
        """
        
        pred_conf = torch.sigmoid(pred[:, 5]) 
        target_conf = target[:, 5] # 1.0 or 0.0

        loss_conf = self.conf_loss(pred_conf, target_conf)
        pos_mask = (target_conf > 0.5) 
        num_pos = pos_mask.sum()
        
        if num_pos > 0:
            loss_reg = self.reg_loss(pred[pos_mask, :4], target[pos_mask, :4])
        else:
            loss_reg = torch.tensor(0.0, device=pred.device, requires_grad=True)
        total_loss = loss_reg + self.lambda_val * loss_conf
        return total_loss
