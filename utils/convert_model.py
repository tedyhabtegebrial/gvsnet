import torch
import torch.nn as nn

# Borrowed from https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/blob/master/sync_batchnorm/batchnorm.py
def convert_model(module):
    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.SynchBatchNorm, to torch.nn.BatchNorm2d
    """
    mod = module
    if isinstance(module, (torch.nn.SyncBatchNorm, )):
        mod = nn.BatchNorm2d(module.num_features, module.eps,
                            module.momentum, module.affine)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))
    return mod
