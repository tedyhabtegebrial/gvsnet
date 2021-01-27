import os
import torch
import torch.nn as nn
from copy import deepcopy
from .sun_model import SUNModel
from .spade import SPADEGenerator, ConvEncoder
from .adn_model import AppearaceDecoderNetwork

from .mpi import ComputeHomography
from .mpi import AlphaComposition
from .mpi import ApplyHomography
from .mpi import Alpha2Disp
from .mpi import ApplyAssociation


class GVSNet(nn.Module):
    def __init__(self, opts):
        super(GVSNet, self).__init__()
        self.opts = opts
        # Appearance Decoder Network
        self.adn = AppearaceDecoderNetwork(opts.feats_per_layer)
        # Semantic Uplifting Network
        self.sun = SUNModel(opts)
        # Scene-style Encoder # from spade
        self.encoder = ConvEncoder(opts)
        # Here we are hacking the SPADE model so that it works as
        # a layered image translator
        spade_ltn_opts = deepcopy(opts)
        spade_ltn_opts.__dict__['num_out_channels'] = opts.num_layers*opts.feats_per_layer
        spade_ltn_opts.__dict__['semantic_nc'] = opts.num_layers * opts.embedding_size
        spade_ltn_opts.__dict__['embedding_size'] = opts.num_layers * opts.embedding_size
        spade_ltn_opts.__dict__['label_nc'] = opts.num_layers*opts.embedding_size
        self.spade_ltn = SPADEGenerator(spade_ltn_opts, no_tanh=True)
        # MPI rendering
        self.compute_homography = ComputeHomography(opts)
        self.alpha_composition = AlphaComposition()
        self.apply_homography = ApplyHomography()
        self.alpha_to_disp = Alpha2Disp(opts)
        self.apply_association = ApplyAssociation(opts.num_layers)

    def _infere_scene_repr(self, input_data):
        layered_sem, mpi_alpha, associations = self.sun(input_data)
        scene_style = self._get_scene_encoding(input_data['style_img'])
        return scene_style, layered_sem, mpi_alpha, associations

    def inference_step(self, input_data):
        assert isinstance(input_data['t_vec'], (list, tuple)), 'a list/tuple of translation vectors should be passed'
        assert isinstance(input_data['r_mat'], (list, tuple)), 'a list/tuple of rotation matrices should be passed'
        color_nv_list, sem_nv_list, disp_nv_list = [], [], []
        with torch.no_grad():
            batch_size = input_data['input_seg'].shape[0]
            num_layers = self.opts.num_layers
            # , self.opts.embedding_size
            height, width = self.opts.height, self.opts.width
            feats_per_layer = self.opts.feats_per_layer
            # Get style encoding of the scene
            # scene_style = self._get_scene_encoding(input_data['style_img'])
            # Infer scene semantics and geometry
            scene_style, layered_sem, mpi_alpha, associations = self._infere_scene_repr(input_data)
            # Render Novel-view semantics and color
            mpi_sem = self.apply_association(layered_sem, input_associations=associations)
            # convert layered semantics to layered appearance
            # mul_layer_sem
            layered_sem = layered_sem.flatten(1, 2)
            # .view(batch_size, num_layers*embedding_size, height, width)
            layered_appearance = self.spade_ltn(layered_sem, z=scene_style).view(batch_size, num_layers, feats_per_layer, height, width)
            # layered appearance to MPI appearance
            mpi_appearance = self.apply_association(
                layered_appearance, input_associations=associations)
            for v in range(len(input_data['t_vec'])):
                k_matrix = input_data['k_matrix']
                t_vec, r_mat = input_data['t_vec'][v], input_data['r_mat'][v]
                # Compute planar homography
                h_mats = self.compute_homography(kmats=k_matrix, r_mats=r_mat, t_vecs=t_vec)
                # Apply homography
                mpi_sem_nv, grid = self.apply_homography(h_matrix=h_mats, src_img=mpi_sem, grid=None)
                mpi_alpha_nv, _ = self.apply_homography(h_matrix=h_mats, src_img=mpi_alpha, grid=grid)
                mpi_app_nv, _ = self.apply_homography(h_matrix=h_mats, src_img=mpi_appearance, grid=grid)
                sem_nv = self.alpha_composition(src_imgs=mpi_sem_nv, alpha_imgs=mpi_alpha_nv)
                if not (self.opts.num_classes==self.opts.embedding_size):
                    sem_nv = self.sem_mpi_net.semantic_embedding.decode(sem_nv)
                # Rendering disparity maps
                disp_nv = self.alpha_to_disp(mpi_alpha_nv, k_matrix, self.opts.stereo_baseline, t_vec, novel_view=True)
                # Rendering Color image
                appearance_nv = self.alpha_composition(
                    src_imgs=mpi_app_nv, alpha_imgs=mpi_alpha_nv)
                # translate appearance features to rgb color space
                color_nv = self.adn(appearance_nv)
                color_nv_list.append(color_nv)
                disp_nv_list.append(disp_nv)
                sem_nv_list.append(sem_nv)
        result_dict = {}
        # result_dict['color_nv']  =  [Num_Cameras, Batch_Size, 3, Height, Width]
        result_dict['color_nv'] = torch.stack(color_nv_list)
        result_dict['disp_nv'] = torch.stack(disp_nv_list)
        result_dict['sem_nv'] = torch.stack(sem_nv_list)
        return result_dict
    
    def training_step(self, input_data):
        raise NotImplementedError
        
    def _get_scene_encoding(self, input_img):
        if not self.opts.use_vae:
            return None
        if self.opts.mode=='train':
            z, mu, logvar = self.encoder(input_img)
            return z, mu, logvar
        else:
            # When we are in test mode unless we explicily want diverse outputs;
            # There is not need to encode the scene and sample from the distribution multiple times
            z, mu, logvar = self.encoder(input_img)
            return mu
