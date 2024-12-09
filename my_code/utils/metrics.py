import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from sklearn.metrics import confusion_matrix, classification_report
"""
Metrics 

# Regression Task
1. MAE # as loss function
2. MSE # as loss function
3. RMSE # as test performance
4. MAPE # as test performance

# Binary Classification Task
1. BCEWithLogitsLoss # as loss function
2. Micro-F1 # test performance
3. Macro-F1 # test performance
"""

# for training
class RegressionLoss(nn.Module):
    def __init__(self, scaler, mask_value=1e-5, loss_type='mae'):
        super(RegressionLoss, self).__init__()
        self._scaler = scaler
        self._mask_value = mask_value
        self._loss_type = loss_type

    def _inv_transform(self, data):
        return self._scaler.inverse_transform(data)

    def _masked_mae(self, preds, labels):
        return torch.mean(torch.abs(preds - labels))

    def _masked_mse(self, preds, labels):
        return torch.mean(torch.square(preds - labels))

    def forward(self, preds, labels):
        # inverse transform
        preds = self._inv_transform(preds)
        labels = self._inv_transform(labels)

        if self._mask_value is not None:
            # mask some elements
            mask = torch.gt(labels, self._mask_value)
            preds = torch.masked_select(preds, mask)
            labels = torch.masked_select(labels, mask)

        if self._loss_type == 'mae':
            return self._masked_mae(preds, labels)
        elif self._loss_type == 'mse':
            return self._masked_mse(preds, labels)
        else:
            raise Exception('Illegal Loss Function\'s Name.')

class LossFusionNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=3):
        super(LossFusionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        # 权重初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, input_size]
        Returns:
            weights: Tensor of shape [batch_size, output_size] where weights sum to 1
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        weights = self.softmax(x)
        return weights

# for testing or validation
class RegressionMetrics(nn.Module):
    def __init__(self, scaler, mask_value=0.0):
        super(RegressionMetrics, self).__init__()
        self._scaler = scaler
        self._mask_value = mask_value

    def _inv_transform(self, data):
        return self._scaler.inverse_transform(data)

    def _masked_rmse(self, preds, labels):
        return torch.sqrt(torch.mean(torch.square(preds - labels)))

    def _masked_mape(self, preds, labels):
        return torch.mean(torch.abs(torch.div((preds - labels), labels)))

    def forward(self, preds, labels):
        # inverse transform
        preds = self._inv_transform(preds)
        labels = self._inv_transform(labels)

        # mask some elements
        mask = torch.gt(labels, self._mask_value)
        preds = torch.masked_select(preds, mask)
        labels = torch.masked_select(labels, mask)

        return (
            self._masked_rmse(preds, labels),
            self._masked_mape(preds, labels)
        )

# need to implement for Binary Classification Task
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2, device=torch.device('cuda:0')):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params).to(device)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum



class ClassificationLoss(nn.Module):
    def __init__(self, pos_weight=None, device=torch.device('cuda:0')):
        super(ClassificationLoss, self).__init__()
        self._pos_weight = pos_weight
        self._device = device

        # 定义各个损失函数
        self.bce_loss = self._BCEWithLogits(pos_weight)
        # 初始化融合网络
        self.loss_fusion = LossFusionNetwork(input_size=3, hidden_size=16, output_size=3).to(device)

    def _BCEWithLogits(self, pos_weight):
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()

    def _Focal_loss(self, logits, targets, alpha=0.8, gamma=3, pos_weight=None):
        bce_loss = self.bce_loss(logits, targets)
        pt = torch.exp(-bce_loss)  # 真实类别的概率
        focal_loss_value = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss_value

    def _Dice_loss(self, logits, targets, smooth=1.0):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)
        intersection = (logits * targets).sum()
        dice = (2. * intersection + smooth) / (logits.sum() + targets.sum() + smooth)
        return 1 - dice

    def _Tversky_loss(self, logits, targets, alpha=0.85, beta=0.15, smooth=1.0):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)
        TP = (logits * targets).sum()
        FP = ((1 - targets) * logits).sum()
        FN = (targets * (1 - logits)).sum()
        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        return 1 - tversky

    def forward(self, preds, labels):
        """
        计算最终的损失值。
        Args:
            preds: 模型的预测值
            labels: 真实标签
        Returns:
            final_loss: 加权后的最终损失
        """
        # 计算各个单独的损失
        loss_focal = self._Focal_loss(preds, labels, pos_weight=self._pos_weight)
        loss_dice = self._Dice_loss(preds, labels)
        loss_tversky = self._Tversky_loss(preds, labels)

        # 计算损失的平均值
        loss_focal_mean = loss_focal.mean().unsqueeze(0)  # [1]
        loss_dice_mean = loss_dice.mean().unsqueeze(0)  # [1]
        loss_tversky_mean = loss_tversky.mean().unsqueeze(0)  # [1]

        # 将损失堆叠成一个向量
        loss_vector = torch.cat([loss_focal_mean, loss_dice_mean, loss_tversky_mean], dim=0)  # [3]
        loss_vector = loss_vector.unsqueeze(0)  # [1, 3]

        # 通过融合网络获取权重
        weights = self.loss_fusion(loss_vector)  # [1, 3]

        # 计算最终的损失
        final_loss = (weights[:, 0] * loss_focal_mean +
                      weights[:, 1] * loss_dice_mean +
                      weights[:, 2] * loss_tversky_mean)

        return final_loss


class ClassificationMetrics(nn.Module):
    def __init__(self, threshold:list):
        super(ClassificationMetrics, self).__init__()
        self._threshold = threshold

    def _round(self, preds):
        preds = torch.sigmoid(preds) # value in [0, 1]
        for idx, threshold in enumerate(self._threshold):
            preds[:, :, :, idx] = torch.where(preds[:, :, :, idx] >= threshold, 1, 0)

        return preds

    def _micro_macro_f1(self, preds, labels):
        # Macro f1
        TP_lst, FN_lst, FP_lst = [], [], []
        # Micro F1
        F1_lst = []
        for c in range(preds.shape[-1]):
            c_preds = preds[:, :, :, c].flatten()
            c_labels = labels[:, :, :, c].flatten()

            TN, FP, FN, TP = confusion_matrix(c_preds, c_labels, labels=[0, 1]).ravel()
            TP_lst.append(TP)
            FN_lst.append(FN)
            FP_lst.append(FP)
            F1_lst.append(2 * TP / (2 * TP + FN + FP))

        macro_f1 = 2 * sum(TP_lst) / (2 * sum(TP_lst) + sum(FN_lst) + sum(FP_lst))
        micro_f1 = sum(F1_lst) / len(F1_lst)

        # ic(classification_report(preds.flatten(), labels.flatten(), labels=np.unique(labels.flatten())))
        # ic(confusion_matrix(preds.flatten(), labels.flatten()))
        return (
            micro_f1,
            macro_f1,
            F1_lst
        )

    def forward(self, preds, labels):
        preds = self._round(preds)

        preds = preds.cpu()
        labels = labels.cpu()

        return self._micro_macro_f1(preds, labels)

