import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu_id', type=int, default=0, help='pass -1 for cpu')
arg_parser.add_argument('--dataset', type=str, default='carla', help='dataset type')
arg_parser.add_argument('--data_path', type=str, default='', help='folder containing the dataset')
arg_parser.add_argument('--height', type=int, default=256)
arg_parser.add_argument('--width', type=int, default=256)
arg_parser.add_argument('--batch_size', type=int, default=1)

arg_parser.add_argument('--num_classes', type=int, default=13)
arg_parser.add_argument('--embedding_size', type=int, default=13)
arg_parser.add_argument('--stereo_baseline', type=float, default=0.24, help='assumed baseline for converting depth to disparity')
arg_parser.add_argument('--style_path', type=str, default='', help='if given the this file will be used as style image')
arg_parser.add_argument('--use_instance_mask', action='store_true', help='is paased, instance mask will be assuned to be present')
arg_parser.add_argument('--mpi_encoder_features', type=int, default=96, help='this controls number feature channels at the output of the base encoder-decoder network')
arg_parser.add_argument('--mode', type=str, default='test', help='choose between [train, test, demo]')

arg_parser.add_argument('--num_layers', type=int, default=3)
arg_parser.add_argument('--feats_per_layer', type=int, default=20)
arg_parser.add_argument('--num_planes', type=int, default=32)
arg_parser.add_argument('--near_plane', type=int, default=1.5, help='nearest plane: 1.5 for carla')
arg_parser.add_argument('--far_plane', type=int, default=20000, help='far plane: 20000 for carla')

######### Arguments for SPADE
arg_parser.add_argument('--d_step_per_g', type=int, default=1, help='num of d updates for each g update')
arg_parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
arg_parser.add_argument('--spade_k_size', type=int,default=3)
arg_parser.add_argument('--num_D', type=int, default=3)
arg_parser.add_argument('--output_nc', type=int, default=3)
arg_parser.add_argument('--n_layers_D', type=int, default=4)

arg_parser.add_argument('--contain_dontcare_label', action='store_true')
arg_parser.add_argument('--no_instance', default=True, type=bool)
arg_parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3', help='instance normalization or batch normalization')
arg_parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
arg_parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')

## Generator settings
arg_parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
arg_parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
arg_parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
arg_parser.add_argument('--z_dim', type=int, default=256, help="dimension of the latent z vector")
## Discriminator setting
arg_parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
arg_parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
arg_parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
arg_parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
arg_parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
arg_parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
arg_parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
arg_parser.add_argument('--lambda_kld', type=float, default=0.001)
arg_parser.add_argument('--num_upsampling_layers',
                    choices=('normal', 'more', 'most'), default='normal',
                    help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
arg_parser.add_argument('--num_out_channels', default=3, type=int)
arg_parser.add_argument('--use_vae', action='store_false') # TODO: This should be updated
arg_parser.add_argument('--aspect_ratio', default=1, type=int)

## Training options
arg_parser.add_argument('--disparity_weight', default=0.1, type=float, 
                        help='for carla=0.1, for other set to 0.5')

