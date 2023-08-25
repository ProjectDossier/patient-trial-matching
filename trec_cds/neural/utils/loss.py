import torch
from torch import nn


class PairwiseHingeLoss(nn.Module):
    def __init__(self):
        super(PairwiseHingeLoss, self).__init__()

    def forward(self, model_predictions_positives, model_predictions_negatives):
        loss = 1.0 - (model_predictions_positives - model_predictions_negatives)
        loss = torch.clamp(loss, min=0.0)
        loss = torch.mean(loss)
        return loss
