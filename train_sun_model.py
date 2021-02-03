import os
import torch
import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from options import arg_parser
from models import SUNModel
from data import get_dataset
from utils import lr_func
from utils import Logger
from utils import dummy_progress_bar

torch.random.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

arg_parser.add_argument('--ngpu', type=int, default=1)
arg_parser.add_argument('--local_rank', type=int, default=0)
opts = arg_parser.parse_args()

# Prepare logging path
os.path.makedirs(opts.logging_path, exist_ok=True)
model_path = os.path.join(opts.logging_path, 'models')
image_path = os.path.join(opts.logging_path, 'images')
os.path.makedirs(model_path, exist_ok=True)
os.path.makedirs(image_path, exist_ok=True)

# Initialize process group
device = f'cuda:{opts.local_rank}'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = 6002
torch.cuda.set_device(opts.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=opts.ngpu,
    rank=opts.local_rank,
)

# Create model
model = SUNModel(opts)
model = model.to(device)
# Wrap model as DDP
model_ddp = DDP(
    model,
    device_ids=[opts.local_rank],
    output_device=opts.local_rank,
    find_unused_parameters=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
optimizer.zero_grad()

dataset = get_dataset(opts.dataset)
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=opts.ngpu,
    rank=opts.local_rank,
    shuffle=True)


data_loader = DataLoader(dataset=dataset,
                          batch_size=opts.batch_size,
                          sampler=sampler,
                          drop_last=True,
                          num_workers=opts.batch_size,
                          pin_memory=True,
                          )



scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func(opts.num_epochs), last_epoch=-1)
logger = Logger()
for epoch in range(opts.num_epochs):
    sampler.set_epoch(epoch)
    with tqdm.tqdm(total=len(data_loader)) if opts.rank == 0 else dummy_progress_bar() as progress_bar:
        for itr, data in tqdm(enumerate(data_loader)):
            data = {k:v.float().to(device) for k,v in data.items()}
            loss_dict, semantics_nv = model_ddp(data, mode='training')
            loss = sum([v for k,v in loss_dict.items()])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Logging loss, predicted images etc
            # if opts.local_rank==0 and False:
            #     logger.log(loss_dict, dtype='scalar')
            #     pred_sem_nv = semantics_nv[0].data.squeeze().cpu()
            #     target_sem_nv = data['target_seg'].squeeze().cpu()
            #     input_sem = data['input_seg'].squeeze().cpu()
            #     logger.log({'pred_sem_novel_v': pred_sem_nv}, dtype='semantics')
            #     logger.log({'real_sem_novel_v': target_sem_nv}, dtype='semantics')
            #     logger.log({'real_sem_input_v': input_sem}, dtype='semantics')            
            if progress_bar:
                progress_bar.set_postfix(disp_loss=loss_dict['disp_loss'],
                                        sem_loss=loss_dict['semantics_loss'],
                                        lr=optimizer.param_groups[0]['lr'])
                progress_bar.update(1)
        # if opts.local_rank == 0 and False:
        #     logger.save_model()
