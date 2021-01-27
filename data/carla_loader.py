import os
import math
import pathlib
import functools
import warnings

import cv2 as cv
from PIL import Image
import numpy as np
from skimage import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CarlaMiniLoader(Dataset):
    '''This class is a loader for a small subset of the carla dataset.
    '''
    def __init__(self, opts):
        super(CarlaMiniLoader, self).__init__()
        data_path = opts.data_path
        self.data_path = data_path
        assert os.path.exists(self.data_path), 'data_path not valid'
        self.width, self.height = opts.width, opts.height
        # The images are captured via stereo pairs
        # The left camera images are in folder HorizontalCameras_00
        # the right camera images are in HorizontalCameras_01.
        sem_imgs = sorted([str(f) for f in pathlib.Path(self.data_path).rglob('*/semantic_segmentation/*')])
        self.left_cam_imgs = [f for f in sem_imgs if 'HorizontalCameras_00' in f]
        self.right_cam_imgs = [f for f in sem_imgs if 'HorizontalCameras_01' in f]
        assert len(self.left_cam_imgs)==len(self.right_cam_imgs), 'Left and right camera images should match in number'
        self.num_classes = opts.num_classes
        self.stereo_baseline = opts.stereo_baseline
        # Incase an outside style path is passed
        self.style_path = opts.style_path
        
    def __len__(self):
        return len(self.left_cam_imgs)

    def __getitem__(self, index):
        # Camera intrinsics
        k_matrix = self.get_k_matrix(height=self.height, width=self.width)
        left_img, right_img = self.left_cam_imgs[index], self.right_cam_imgs[index]
        # Load semantic maps
        input_sem = self.read_semantics(left_img)
        target_sem = self.read_semantics(right_img)
        # Color
        input_col = self.read_color(left_img)
        target_col = self.read_color(right_img)
        # Disparity
        input_disp = self.read_disparity(left_img, k_matrix)
        target_disp = self.read_disparity(right_img, k_matrix)
        # Camera pose
        t_vec = torch.FloatTensor([-self.stereo_baseline, 0, 0]).view(3, 1)
        r_mat = torch.eye(3)
        data_dict = {}
        data_dict['input_img'] = input_col
        data_dict['input_seg'] = input_sem
        data_dict['input_disp'] = input_disp
        data_dict['target_img'] = target_col
        data_dict['target_seg'] = target_sem
        data_dict['target_disp'] = target_disp
        data_dict['k_matrix'] = k_matrix
        data_dict['t_vec'] = t_vec
        data_dict['r_mat'] = r_mat
        data_dict['stereo_baseline'] = torch.Tensor([self.stereo_baseline])
        # Load style image, if passed, else the input will serve as style
        if self.style_path=='':
            data_dict['style_img'] = input_col.clone()
        else:
            data_dict['style_img'] = self.read_color(self.style_path, map_name=False)
        # 
        # if any of the disctionary items are None we remove its them
        data_dict = {k:v.float() for k,v in data_dict.items() if not (k is None)}
        return data_dict

    def read_color(self, filename, map_name=True):
        filename = filename.replace('semantic_segmentation', 'rgb') if map_name else filename
        try:
            img = io.imread(filename)[..., :3]
        except:
            print(f'cannot read {filename}')
            return None
        img_np = cv.resize(img, (self.width, self.height)) / 255.0
        img_np = (2*img_np)-1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor

    def read_semantics(self, filename):
        # Read semantic map
        try:
            sem_img = cv.imread(filename, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        except:
            print(f'cannot read {filename}')
            return None
        sem_img = np.asarray(sem_img, dtype=np.uint8)
        sem_img = torch.from_numpy(sem_img[..., 2]).squeeze()
        assert sem_img.max()<self.num_classes, 'semantic ids should be <= num_classes-1 '
        sem_img = F.interpolate(input=sem_img.unsqueeze(0).unsqueeze(0), size=(self.height, self.width), mode='nearest').squeeze(0)
        # Convert label to one hot
        labels = torch.zeros(self.num_classes, self.height, self.width)
        labels = labels.scatter_(dim=0, index=sem_img.long(), value=1.0)
        return labels

    def read_disparity(self, filename, kmats):
        try:
            depth_img = np.asarray(Image.open(filename.replace('semantic_segmentation', 'depth')), dtype=np.uint8)
        except:
            print(f'cannot read {filename}')
            return None
        depth_img = depth_img.astype(np.float64)[:,:,:3]
        normalized_depth = np.dot(depth_img, [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        normalized_depth = torch.from_numpy(normalized_depth * 1000.0)
        disparity = self.stereo_baseline * kmats[0,0] / (normalized_depth + 0.0000000001)
        disparity = F.interpolate(disparity.unsqueeze(0).unsqueeze(0), size=(self.height, self.width), mode='bilinear', align_corners=False).squeeze(0)
        return disparity

    def get_k_matrix(self, fov=90.0, height=600, width=800):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / \
                            (2.0 * math.tan(fov * math.pi / 360.0))
        return torch.from_numpy(k)

class LoadSamples(Dataset):
    def __init__(self, opts):
        super(LoadSamples, self).__init__()
        self.opts = opts
        assert os.path.exists(self.opts.data_path)
        files = sorted([f.__str__() for f in pathlib.Path(self.opts.data_path).rglob('*.png')])
        sem_files = [f for f in files if f.endswith('semantics.png')]
        assert len(
            sem_files) > 0, f'It seems like the folder {self.opts.data_path} contains no semantic maps'
        self.sem_files = sem_files
        col_files = [f for f in files if 'color.png' in f]
        self.col_files = col_files if len(col_files)>0 else None
        
        if opts.dataset == 'carla':
            self.reader = CarlaMiniLoader(opts)

    def __getitem__(self, index):
        data_dict = {}
        data_dict['input_seg'] = self.read_semantics(self.sem_files[index])
        if os.path.exists(self.opts.style_path):
            # we will use the external image as style
            data_dict['style_img']  = self.read_color(self.opts.style_path)
        else:
            data_dict['style_img'] = self.read_color(self.col_files[index])
        data_dict['k_matrix'] = self.get_k_matrix(height=self.opts.height, width=self.opts.width)
        data_dict = {k:v.float() for k,v in data_dict.items()}
        return data_dict

    def __len__(self):
        return len(self.sem_files)

    def read_semantics(self, filename):
        sem_img = np.array(Image.open(filename).convert(
            mode="P"), dtype=np.uint8)
        sem_img = torch.from_numpy(sem_img).squeeze()
        assert sem_img.max() < self.opts.num_classes, 'semantic ids should be <= num_classes-1 '
        sem_img = F.interpolate(input=sem_img.float().unsqueeze(0).unsqueeze(
            0), size=(self.opts.height, self.opts.width), mode='nearest').squeeze(0)
        # Convert label to one hot
        labels = torch.zeros(self.opts.num_classes,
                             self.opts.height, self.opts.width)
        labels = labels.scatter_(dim=0, index=sem_img.long(), value=1.0)
        return labels

    def read_color(self, filename):
        img = io.imread(filename)[..., :3]
        img_np = cv.resize(img, (self.opts.width, self.opts.height)) / 255.0
        img_np = (2*img_np)-1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor

    def get_k_matrix(self, fov=90.0, height=600, width=800):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / \
            (2.0 * math.tan(fov * math.pi / 360.0))
        return torch.from_numpy(k)
