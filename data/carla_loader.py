import os
import math
import random
from PIL import Image
from pathlib import Path
import cv2 as cv
from skimage import io
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset


class Carla(Dataset):

    def __init__(self, opts):
        super(Carla, self).__init__()
        self.opts = opts
        # Transformations
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.ToPIL = transforms.Compose([transforms.ToPILImage()])       

        self.height, self.width = self.opts.height, self.opts.width
        self.stereo_baseline = 0.54

        self.base_path = f'{opts.data_path}/{opts.mode}'
        assert os.path.exists(
            self.base_path), f'{self.base_path} folder doesn"t exist'
        self.file_list = self.get_file_list()
        # 
        self.train_camera_suffix = [f'{str(x).fill(2)}' for x in 5]

    def get_file_list(self):
        if self.mode == 'train':
            ''' For training we load a single view from a random camera group, town, weather condition and time step.
            A sample file in the returned list would look like
                 .../Town04/weather_03/HorizontalCameras/rgb/009990.png
            The source and target cameras will be decided inside the __getitem__()
            Example source and target views could be:
                .../Town00/weather_03/HorizontalCameras_03/rgb/009990.png
                .../Town00/weather_03/HorizontalCameras_00/rgb/009990.png
            '''
            episode_folders = [
                f'Town0{x}/weather_0{y}' for x in range(1, 6) for y in range(4)]
            camera_groups = ['ForwardCameras','SideCameras', 'HorizontalCameras']
            join = lambda x:os.path.join(a for a in x)
            file_list = [join([epi, cam, f'rgb/{str(x).zfill(6)}.png']) for epi in episode_folders
                            for cam in camera_groups \
                                for x in range(0, 10000, 10)]
            return file_list
        else:
            ''' For test we load list of source and target views
            In test phase the reference camera is at 00 and target it at 01 
            Example source and target views could be:
                .../Town04/weather_03/HorizontalCameras_00/rgb/000000.png
                .../Town04/weather_03/HorizontalCameras_01/rgb/000000.png
            '''
            test_frames = []
            with open(os.path.join(os.getcwd(), 'data/carla_test_frames.txt'), 'r') as fid:
                reader = csv.reader(fid)
                for line in reader:
                    src = os.path.join(self.base_path, line[0])
                    trg = os.path.join(self.base_path, line[1])
                    test_frames.append([src, trg])
            return test_frames

    def __getitem__(self, index):
        if self.mode=='train':
            sample = self.file_list[index]
            trg_cam, src_cam = random.sample(self.train_camera_suffix, 2)
            cam_group = Path(sample).parent.parent.stem
            src_file = sample.replace(cam_group, cam_group+src_cam)
            trg_file = sample.replace(cam_group, cam_group+trg_cam)
        else:
            src_file, trg_file = self.file_list[index][0], self.file_list[index][1]
        input_img = self._read_rgb(src_file)
        target_img = self._read_rgb(trg_file)
        k_matrix = self._carla_k_matrix(self.height, self.width)
        input_disp = self._read_disp(src_file.replace('rgb', 'depth'), k_matrix)
        target_disp = self._read_disp(trg_file.replace('rgb', 'depth'), k_matrix)
        input_seg = self._read_disp(src_file.replace('rgb', 'semantic_segmentation'), k_matrix)
        target_seg = self._read_disp(trg_file.replace(
            'rgb', 'semantic_segmentation'), k_matrix)
        r_mat, t_vec = self._get_rel_pose(src_file, trg_file)
        data_dict = {}
        data_dict['input_img'] = input_img
        data_dict['input_seg'] = input_seg
        data_dict['input_disp'] = input_disp
        data_dict['target_img'] = target_img
        data_dict['target_seg'] = target_seg
        data_dict['target_disp'] = target_disp
        data_dict['k_matrix'] = k_matrix
        data_dict['t_vec'] = t_vec
        data_dict['r_mat'] = r_mat
        data_dict['stereo_baseline'] = torch.Tensor([self.stereo_baseline])
        # Load style image, if passed, else the input will serve as style
        data_dict['style_img'] = input_img.clone()
        data_dict = {k: v.float()
                     for k, v in data_dict.items() if not (k is None)}
        return data_dict
        
    def _get_rel_pose(self, src_file, trg_file):
        cam_src = Path(src_file).parent.parent.stem
        cam_trg = Path(src_file).parent.parent.stem
        src_idx, trg_idx = int(cam_src[-2:]), int(cam_trg[-2:])
        if cam_src.startswith('ForwardCameras'):
            x, y = 0, 0
            z = (src_idx - trg_idx)*self.stereo_baseline
        elif cam_src.startswith('HorizontalCameras'):
            y, z = 0, 0
            x = (src_idx - trg_idx)*self.stereo_baseline
        elif cam_src.startswith('SideCameras'):
            y, z = 0, 0
            x = (trg_idx - src_idx)*self.stereo_baseline
        else:
            assert False, f'unknown camera identifier {cam_src}'

        t_vec = torch.FloatTensor([x, y, z]).view(3, 1)
        r_mat = torch.eye(3).float()
        return r_mat, t_vec

    def _read_depth(self, depth_path):
        img = np.asarray(Image.open(depth_path), dtype=np.uint8)
        img = img.astype(np.float64)[:,:,:3]
        normalized_depth = np.dot(img, [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        normalized_depth = torch.from_numpy(normalized_depth * 1000.0)
        return normalized_depth

    def _read_disp(self, depth_path, k_matrix):
        depth_img = self._read_depth(depth_path).squeeze()
        disp_img = self.stereo_baseline * \
            k_matrix / (depth_img.clamp(min=1e-06)).squeeze()
        h, w = disp_img.shape[:2]
        if h!=self.height or w!=self.width:
            disp_img = disp_img.view(1, 1, h, w)
            disp_img = F.interpolate(disp_img, size=(self.height, self.width), 
                                    mode='bilinear', align_corners=False)
        disp_img = disp_img.view(1, self.height, self.width)
        return disp_img

    def __len__(self):
        return len(self.file_list)

    def label_to_one_hot(self, input_seg, num_classes=13):
        assert input_seg.max() < num_classes, f'Num classes == {input_seg.max()} exceeds {num_classes}'
        b, _, h, w = input_seg.shape
        lables = torch.zeros(b, num_classes, h, w).float()
        labels = lables.scatter_(dim=1, index=input_seg.long(), value=1.0)
        labels = labels.to(input_seg.device)
        return labels

    def _read_seg(self, semantics_path):
        seg = cv.imread(semantics_path, cv.IMREAD_ANYCOLOR |
                        cv.IMREAD_ANYDEPTH)
        seg = np.asarray(seg, dtype=np.uint8)
        seg = torch.from_numpy(seg[..., 2]).float().squeeze()
        h, w = seg.shape
        seg = F.interpolate(seg.view(1, 1, h, w), size=(self.height, self.width),
                            mode='nearest')
        # Change semantic labels to one-hot vectors
        seg = self.label_to_one_hot(seg, self.opts.num_classes).squeeze(0)
        return seg

    def _carla_k_matrix(self, fov=90.0, height=256, width=256):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / \
            (2.0 * math.tan(fov * math.pi / 360.0))
        return torch.from_numpy(k)

    def _read_rgb(self, img_path):
        img = io.imread(str(img_path))
        img = img[:, :, :3]
        img = cv.resize(img, (self.width, self.height)) / 255.0
        img = (2*img)-1.0
        img_tensor = torch.from_numpy(img).transpose(
            2, 1).transpose(1, 0).float()
        return img_tensor