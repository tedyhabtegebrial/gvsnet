import torch
import torch.nn as nn
import torch.nn.functional as F

class ApplyAssociation(nn.Module):
    def __init__(self, num_layers):
        super(ApplyAssociation, self).__init__()

    def forward(self, input_features, input_associations):
        feat, assoc = input_features, input_associations
        assoc_safe = assoc + 1e-06
        assoc_normalised = assoc_safe/torch.sum(assoc_safe, dim=2, keepdim=True)
        associated_features = torch.mul(feat.unsqueeze(1), assoc_normalised.unsqueeze(3)).sum(dim=2)
        return associated_features
