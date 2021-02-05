import os
import torch
import random
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch import distributed as dist
from itertools import chain

from options import arg_parser
from models import GVSNet
from data import get_dataset
from utils import lr_func
from utils import Logger
from utils import dummy_progress_bar

torch.random.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

arg_parser.add_argument('--ngpu', type=int, default=1)
arg_parser.add_argument('--local_rank', type=int, default=0)
# arg_parser.add_argument('--port', type=int, default=8008)
opts = arg_parser.parse_args()

# Initialize process group
if opts.slurm:
    def init_process_group():
        # os.environ['NCCL_IB_DISABLE'] = '1'
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        rank = int(os.getenv('RANK', 0))
        world_size = int(os.getenv('WORLD_SIZE', 1))
        num_nodes = int(os.getenv('SLURM_NNODES', 1))
        print(f'Global Rank {rank} | Local rank {local_rank} | world size {world_size}')
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        return rank, world_size, num_nodes
    rank, world_size, num_nodes = init_process_group()
    opts.__dict__['local_rank'] = rank
    opts.__dict__['world_size'] = world_size
    device = f'cuda:{opts.local_rank}'
    torch.cuda.set_device(opts.local_rank)
else:
    device = f'cuda:{opts.local_rank}'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{opts.port}'
    # Launch processes
    torch.cuda.set_device(opts.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=opts.ngpu,
        rank=opts.local_rank,
    )
# Prepare logging path
if opts.local_rank == 0:
    print(f'Find tensorboard at {opts.logging_path}')
    os.makedirs(opts.logging_path, exist_ok=True)
    model_path = os.path.join(opts.logging_path, 'models')
    image_path = os.path.join(opts.logging_path, 'images')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
# Create model
model = GVSNet(opts)
model.sun.load_state_dict(torch.load(opts.pre_trained_sun))
model = model.to(device)

dataset = get_dataset(opts.dataset)(opts)
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=opts.world_size if opts.slurm else opts.ngpu,
    rank=opts.local_rank,
    shuffle=True)

data_loader = DataLoader(dataset=dataset,
                         batch_size=opts.batch_size,
                         sampler=sampler,
                         drop_last=True,
                         num_workers=opts.batch_size,
                         pin_memory=True,
                         )

gen_param_list = []
gen_param_list.extend([model.adn.parameters()])
gen_param_list.extend([model.encoder.parameters()])
gen_param_list.extend([model.spade_ltn.parameters()])

gen_optimizer = torch.optim.Adam(
    chain(*gen_param_list), lr=opts.gen_lr, betas=(0.9, 0.999))

disc_optimizer = torch.optim.Adam(model.discriminator.parameters(),
                                  lr=opts.disc_lr, betas=(0, 0.999))


disc_scheduler = torch.optim.lr_scheduler.LambdaLR(
    disc_optimizer, lr_lambda=lr_func(opts.num_epochs))
gen_scheduler = torch.optim.lr_scheduler.LambdaLR(
    gen_optimizer, lr_lambda=lr_func(opts.num_epochs))


if opts.slurm:
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device],
        find_unused_parameters=True,)
else:
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        find_unused_parameters=True,)

logger = Logger(opts.logging_path) if opts.local_rank == 0 else None
iter_count = 0
for epoch in range(opts.num_epochs):
    sampler.set_epoch(epoch)
    if logger:
        logger.save_model(model, epoch)
    with tqdm.tqdm(total=len(data_loader)) if opts.local_rank == 0 else dummy_progress_bar() as progress_bar:
        for itr, data in enumerate(data_loader):
            data = {k: v.float().to(device) for k, v in data.items()}
            gen_optimizer.zero_grad()
            gen_loss_dict = model_ddp(data, mode='generator')
            g_loss = sum(
                [v for k, v in gen_loss_dict.items() if not v is None])
            g_loss.backward()
            gen_optimizer.step()
            # disc_losses
            disc_optimizer.zero_grad()
            disc_loss_dict = model_ddp(data, mode='discriminator')
            d_loss = sum([v for k, v in disc_loss_dict.items() if not v is None])
            d_loss.backward()
            disc_optimizer.step()            
            # Logging loss, predicted images etc
            if logger:
                logger.log_scalar(gen_loss_dict)
                logger.log_scalar(disc_loss_dict)
                if itr % opts.image_log_interval == 0:
                    input_sem = data['input_seg'].squeeze().cpu()
                    logger.save_semantics({'sem_gt_input_v': input_sem[0]})
                    logger.save_images(
                        {'rgb_gt_input_v': data['input_img'][0].cpu()})
                    logger.save_images(
                        {'rgb_gt_novel_v': data['input_img'][0].cpu()})
                    logger.save_images(
                        {'rgb_pred_novel_v': model_ddp.module.fake[0].cpu()})
            if progress_bar:
                progress_bar.set_postfix(d_loss={k: f'{v.item():.3f}' for k, v in disc_loss_dict.items() if not v is None},
                                         g_loss={k: f'{v.item():.3f}' for k, v in gen_loss_dict.items() if not v is None},
                                        g_lr=gen_optimizer.param_groups[0]['lr'],
                                        d_lr=disc_optimizer.param_groups[0]['lr'],)
                progress_bar.update(1)
            if logger:
                logger.step()
            if iter_count<1000 or iter_count%1000==0:
                gen_scheduler.step()
                disc_scheduler.step()
            iter_count += 1
