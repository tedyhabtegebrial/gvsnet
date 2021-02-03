import os
import torch
import tqdm
from torch.utils.data import DataLoader

from options import arg_parser
from models import GVSNet
from data import get_dataset
from utils import SaveResults
from utils import get_current_time
from utils import get_cam_poses
from utils import convert_model


arg_parser.add_argument('--movement_type', default='circle', choices=['circle', 'horizontal', 'forward'], 
                        help='camera movement type: ')
arg_parser.add_argument('--output_path', type=str, default=f'./output/{get_current_time()}', 
                        help='path for saving results')



opts = arg_parser.parse_args()
device = f'cuda:{opts.gpu_id}' if opts.gpu_id>-1 else 'cpu'

# prapare folder for results
os.makedirs(opts.output_path, exist_ok=True)

gvs_net = GVSNet(opts)
gvs_net.load_state_dict(torch.load(opts.pre_trained_gvsnet), strict=True)
gvs_net.to(device)
gvs_net.eval()

if device=='cpu':
    gvs_net = convert_model(gvs_net)
dataset = DataLoader(get_dataset(opts.dataset)(opts),
                     batch_size=1, shuffle=False)

saver_results = SaveResults(opts.output_path, opts.dataset)

for itr, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
    data = {k:v.to(device) for k,v in data.items()}
    # Let's get a list of camera poses
    # modify get_cam_poses function if you need specific camera movement
    data['t_vec'], data['r_mat'] = get_cam_poses(opts.movement_type, b_size=opts.batch_size)
    data['t_vec'] = [t.to(device) for t in data['t_vec']]
    data['r_mat'] = [r.to(device) for r in data['r_mat']]
    # Render the scene from the chosen camera poses
    results_dict = gvs_net.render_multiple_cams(data)
    saver_results(results_dict, itr)
print(f'Find results at {opts.output_path}')
