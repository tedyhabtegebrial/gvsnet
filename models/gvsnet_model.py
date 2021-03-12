import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from .sun_model import SUNModel
from .spade import SPADEGenerator, ConvEncoder
from .adn_model import AppearaceDecoderNetwork

from .mpi import ComputeHomography
from .mpi import AlphaComposition
from .mpi import ApplyHomography
from .mpi import Alpha2Disp
from .mpi import ApplyAssociation

from .spade import VGGLoss
from .spade import KLDLoss
from .spade import GANLoss
from .spade import MultiscaleDiscriminator


class GVSNet(nn.Module):
    def __init__(self, opts):
        super(GVSNet, self).__init__()
        self.opts = opts
        # Appearance Decoder Network
        self.adn = AppearaceDecoderNetwork(opts.feats_per_layer)
        # Semantic Uplifting Network
        self.sun = SUNModel(opts)
        if opts.mode == 'train':
            # We assume the semantic uplifiting network is already trained
            self.sun.eval()
        # Scene-style Encoder # from spade
        self.encoder = ConvEncoder(opts)
        # Here we are hacking the SPADE model so that it works as
        # a layered image translator
        spade_ltn_opts = deepcopy(opts)
        spade_ltn_opts.__dict__[
            'num_out_channels'] = opts.num_layers * opts.feats_per_layer
        spade_ltn_opts.__dict__[
            'semantic_nc'] = opts.num_layers * opts.embedding_size
        spade_ltn_opts.__dict__[
            'embedding_size'] = opts.num_layers * opts.embedding_size
        spade_ltn_opts.__dict__[
            'label_nc'] = opts.num_layers * opts.embedding_size
        self.spade_ltn = SPADEGenerator(spade_ltn_opts, no_tanh=True)
        if opts.mode == 'train':
            # Discriminator
            self.discriminator = MultiscaleDiscriminator(opts)
        # MPI rendering
        self.compute_homography = ComputeHomography(opts)
        self.alpha_composition = AlphaComposition()
        self.apply_homography = ApplyHomography()
        self.alpha_to_disp = Alpha2Disp(opts)
        self.apply_association = ApplyAssociation(opts.num_layers)
        if opts.mode == 'train':
            self.vgg_loss = VGGLoss()
        self.get_kld_loss = KLDLoss()
        self.get_gan_loss = GANLoss(opts.gan_mode, opts)

    def _infere_scene_repr(self, input_data):
        layered_sem, mpi_alpha, associations = self.sun(input_data)
        scene_style = self._get_scene_encoding(input_data['style_img'])
        return scene_style, layered_sem, mpi_alpha, associations

    def render_multiple_cams(self, input_data):
        assert isinstance(input_data['t_vec'], (list, tuple)
                          ), 'a list/tuple of translation vectors should be passed'
        assert isinstance(input_data['r_mat'], (list, tuple)
                          ), 'a list/tuple of rotation matrices should be passed'
        color_nv_list, sem_nv_list, disp_nv_list = [], [], []
        with torch.no_grad():
            batch_size = input_data['input_seg'].shape[0]
            num_layers = self.opts.num_layers
            height, width = self.opts.height, self.opts.width
            feats_per_layer = self.opts.feats_per_layer
            # Infer scene semantics and geometry
            scene_style, layered_sem, mpi_alpha, associations = self._infere_scene_repr(
                input_data)

            # Render Novel-view semantics and color
            mpi_sem = self.apply_association(
                layered_sem, input_associations=associations)
            layered_sem = layered_sem.flatten(1, 2)
            layered_appearance = self.spade_ltn(layered_sem, z=scene_style[1]).view(
                batch_size, num_layers, feats_per_layer, height, width)
            mpi_appearance = self.apply_association(
                layered_appearance, input_associations=associations)
            for v in range(len(input_data['t_vec'])):
                k_matrix = input_data['k_matrix']
                t_vec, r_mat = input_data['t_vec'][v], input_data['r_mat'][v]
                # Compute planar homography
                h_mats = self.compute_homography(
                    kmats=k_matrix, r_mats=r_mat, t_vecs=t_vec)
                # Apply homography
                mpi_sem_nv, grid = self.apply_homography(
                    h_matrix=h_mats, src_img=mpi_sem, grid=None)
                mpi_alpha_nv, _ = self.apply_homography(
                    h_matrix=h_mats, src_img=mpi_alpha, grid=grid)
                mpi_app_nv, _ = self.apply_homography(
                    h_matrix=h_mats, src_img=mpi_appearance, grid=grid)
                sem_nv = self.alpha_composition(
                    src_imgs=mpi_sem_nv, alpha_imgs=mpi_alpha_nv)
                if not (self.opts.num_classes == self.opts.embedding_size):
                    sem_nv = self.sem_mpi_net.semantic_embedding.decode(sem_nv)
                # Rendering disparity maps
                disp_nv = self.alpha_to_disp(
                    mpi_alpha_nv, k_matrix, self.opts.stereo_baseline, t_vec, novel_view=True)
                # Rendering Color image
                appearance_nv = self.alpha_composition(
                    src_imgs=mpi_app_nv, alpha_imgs=mpi_alpha_nv)
                # translate appearance features to rgb color space
                color_nv = self.adn(appearance_nv)
                color_nv_list.append(color_nv)
                disp_nv_list.append(disp_nv)
                sem_nv_list.append(sem_nv)
        result_dict = {}
        result_dict['color_nv'] = torch.stack(color_nv_list)
        result_dict['disp_nv'] = torch.stack(disp_nv_list)
        result_dict['sem_nv'] = torch.stack(sem_nv_list)
        return result_dict

    def forward(self, input_data, mode='generator'):
        if mode == 'generator':
            color_nv, kld_loss = self.generate_fake(input_data)
            gen_losses = self.compute_generator_loss(
                color_nv, input_data['target_img'], input_data['target_seg'])
            gen_losses['kld_loss'] = kld_loss
            self.real = input_data['target_img']
            self.fake = color_nv.data
            return gen_losses
        elif mode == 'discriminator':
            disc_losses = self.compute_discriminator_loss(input_data)
            return disc_losses
        elif mode == 'inference':
            with torch.no_grad():
                color_nv, _kld_loss = self.generate_fake(input_data)
            return color_nv
        else:
            raise KeyError('')

    def _get_scene_encoding(self, input_img):
        if not self.opts.use_vae:
            return None, None, None
        if self.opts.mode == 'train':
            z, mu, logvar = self.encoder(input_img)
            return z, mu, logvar
        else:
            # When we are in test mode unless we explicily want diverse outputs;
            # There is not need to encode the scene and sample from the distribution multiple times
            z, mu, logvar = self.encoder(input_img)
            return None, mu, None

    def generate_fake(self, input_data):
        num_layers = self.opts.num_layers
        height, width = self.opts.height, self.opts.width
        feats_per_layer = self.opts.feats_per_layer
        batch_size = input_data['input_seg'].shape[0]
        # Get style encoding of the scene
        z, mu, logvar = self._get_scene_encoding(input_data['style_img'])
        # Infer scene semantics and geometry
        with torch.no_grad():
            layered_sem, mpi_alpha, associations = self.sun(input_data)
            #layered_sem = F.softmax(layered_sem, dim=2)
        layered_sem = layered_sem.flatten(1, 2)
        layered_appearance = self.spade_ltn(layered_sem, z=z).view(
            batch_size, num_layers, feats_per_layer, height, width)
        mpi_appearance = self.apply_association(
            layered_appearance, input_associations=associations)
        # Here we do novel-view synthesis of apearance features
        t_vec, r_mat = input_data['t_vec'], input_data['r_mat']
        # Compute planar homography
        h_mats = self.compute_homography(
            kmats=input_data['k_matrix'], r_mats=r_mat, t_vecs=t_vec)
        mpi_alpha_nv, grid = self.apply_homography(
            h_matrix=h_mats, src_img=mpi_alpha)
        mpi_app_nv, _ = self.apply_homography(
            h_matrix=h_mats, src_img=mpi_appearance, grid=grid)
        appearance_nv = self.alpha_composition(
            src_imgs=mpi_app_nv, alpha_imgs=mpi_alpha_nv)
        color_nv = self.adn(appearance_nv)
        kld_loss = self.get_kld_loss(mu, logvar) * self.opts.lambda_kld
        return color_nv, kld_loss

    def compute_generator_loss(self, fake_image, real_img, real_seg):
        device_ = fake_image.device
        gen_losses = {}
        if not self.opts.embedding_size == self.opts.num_classes:
            real_seg = self.sem_mpi_net.semantic_embedding.encode(real_seg)
        pred_fake, pred_real = self.discriminate(
            real_seg, fake_image, real_img)
        gen_losses['GAN'] = sum(self.get_gan_loss(
            pred_fake, True, for_discriminator=False))
        if not self.opts.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(device_)
            for i in range(num_D):
                # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = F.l1_loss(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opts.lambda_feat / num_D
            gen_losses['GAN_Feat'] = sum(GAN_Feat_loss)
        if not self.opts.no_vgg_loss:
            gen_losses['VGG'] = self.vgg_loss(
                fake_image, real_img) * self.opts.lambda_vgg
        return gen_losses

    def compute_discriminator_loss(self, input_data):
        real_seg = input_data['target_seg']
        real_img = input_data['target_img']
        D_losses = {}
        with torch.no_grad():
            fake_image, _kld_loss = self.generate_fake(input_data)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        if not self.opts.embedding_size == self.opts.num_classes:
            real_seg = self.sem_mpi_net.semantic_embedding.encode(real_seg)
        pred_fake, pred_real = self.discriminate(
            real_seg, fake_image, real_img)

        D_losses['D_Fake'] = self.get_gan_loss(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.get_gan_loss(pred_real, True,
                                               for_discriminator=True)
        return D_losses

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.discriminator(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
