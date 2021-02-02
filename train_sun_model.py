import os
import torch
import random
import numpy as np

from options import arg_parser
from models import GVSNet

torch.random.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

opts = arg_parser.parse_args()

