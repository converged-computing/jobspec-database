import torch
import argparse
import sys
import pickle as pkl

from nerf.provider import NeRFDataset
from nerf.utils import *
from nerf.network_particle import NeRFNetwork

import dnnlib

IMAGE_RES = 256

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = None # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = IMAGE_RES # Be explicit about resolution.
        dataset_kwargs.use_labels = True # Be explicit about labels.
        dataset_kwargs.max_size = 99999999 # Be explicit about dataset size.
        return dataset_kwargs, 'place_holder'
    except IOError as err:
        raise ValueError(f'--data: {err}')

def make_c(opt):
    # print(opt)
    opts = dnnlib.EasyDict(vars(opt))
    
    c = dnnlib.EasyDict()
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise ValueError('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    # c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    # c.num_gpus = opts.gpus
    # c.batch_size = opts.batch
    # c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    # c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    # c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    # c.loss_kwargs.r1_gamma = opts.gamma
    # c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    # c.D_opt_kwargs.lr = opts.dlr
    # c.metrics = opts.metrics
    # c.total_kimg = opts.kimg
    # c.kimg_per_tick = opts.tick
    # c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    # c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    # c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    # if c.batch_size % c.num_gpus != 0:
    #     raise click.ClickException('--batch must be a multiple of --gpus')
    # if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
    #     raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    # if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
    #     raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    # if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
    #     raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    # c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = 'training.triplane.TriPlaneGenerator'
    c.D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning

    if c.training_set_kwargs.resolution == 512:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
    elif c.training_set_kwargs.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"
    
    if opts.sr_module != None:
        sr_module = opts.sr_module
    
    rendering_options = {
        'image_resolution': c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not opts.gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale': opts.c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': opts.sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg, # strength of density regularization
        'density_reg_p_dist': opts.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': opts.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': opts.decoder_lr_mul, # learning rate multiplier for decoder
        'sr_antialias': True,
    }

    if opts.cfg == 'ffhq':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 'auto', #'2.25', # near point along each ray to start taking samples.
            'ray_end': 'auto', #3.3, # far point along each ray to stop taking samples. 
            'box_warp': 2, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.], # used only in the visualizer to control center of camera rotation.
            'white_back': True,
        })
    elif opts.cfg == 'afhq':
        rendering_options.update({
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 'auto',  # 2.25,
            'ray_end': 'auto', #3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    elif opts.cfg == 'shapenet':
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            'ray_start': 'auto', #0.1,
            'ray_end': 'auto', #2.6,
            'box_warp': 1.6,
            'white_back': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
        })
    else:
        assert False, "Need to specify config"



    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.num_fp16_res = 0
    # c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    # c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = True
    c.loss_kwargs.neural_rendering_resolution_initial = opts.neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = opts.neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = opts.neural_rendering_resolution_fade_kimg
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, fused_modconv_default='inference_only')

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    # Augmentation.
    # if opts.aug != 'noaug':
    #     c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
    #     if opts.aug == 'ada':
    #         c.ada_target = opts.target
    #     if opts.aug == 'fixed':
    #         c.augment_p = opts.p

    # Resume.
    # if opts.resume is not None:
    #     c.resume_pkl = opts.resume
    #     c.ada_kimg = 100 # Make ADA react faster at the beginning.
    #     c.ema_rampup = None # Disable EMA rampup.
    #     if not opts.resume_blur:
    #         c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
    #         c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    # if opts.nobench:
    #     c.cudnn_benchmark = False
        
    # print('built c!')
    return c 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=5, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--test_interval', type=int, default=5000, help="evaluate on the test set every interval epochs")
    parser.add_argument('--workspace', type=str, default='exp/')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', default=None)

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=1e5, help="target face number for mesh decimation")

    parser.add_argument('--dmtet', action='store_true', help="use dmtet")
    parser.add_argument('--tet_grid_size', type=int, default=256, help="tet grid size")
    parser.add_argument('--init_ckpt', type=str, default='', help="ckpt to init dmtet")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--warm_iters', type=int, default=500, help="training iters")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo', action='store_true', default=True, help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='softplus', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=0.1, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=10, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.5, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='particle', choices=['grid', 'vanilla', 'particle'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adam', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=512, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=512, help="render height for NeRF in training")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.6, 1.8], help="training camera radius range")
    parser.add_argument('--val_radius', type=float, default=3.0, help="valid camera radius")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true', default=True, help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--val_theta', type=float, default=60, help="Angle when validating")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[0, 120], help="training camera up-down theta range")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")

    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=10, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_lap', type=float, default=0.5, help="loss scale for mesh laplacian")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--tri_res', type=int, default=64, help="resolution of triple plane")
    parser.add_argument('--num_layers', type=int, default=1, help="num layers of MLP decoder")
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dims of MLP decoder")
    parser.add_argument('--decoder_act', type=str, default="relu", choices=["relu", "softplus"], help="hidden dims of MLP decoder")
    parser.add_argument('--per_iter', type=int, default=100, help="iters per epoch")

    parser.add_argument('--K', type=int, default=1, help="K unet iters per particle optimization iters")
    parser.add_argument('--K2', type=int, default=1, help="1 unet iters per K2 iters")

    parser.add_argument('--unet_bs', type=int, default=1, help="batch size of unet")
    parser.add_argument('--unet_lr', type=float, default=0.0001, help="learning rate of unet")
    parser.add_argument('--val_size', type=int, default=7, help="size of val set")
    parser.add_argument('--val_nz', type=int, default=5, help="number of z of val set")
    parser.add_argument('--scale', type=float, default=100, help="guidance scale")

    parser.add_argument('--q_iter', type=int, default=0, help="time to start using q")
    parser.add_argument('--q_rate', type=float, default=1, help="strength of H(q) loss")
    parser.add_argument('--latent', type=bool, default=False, help="wheather to render in latent mode")
    parser.add_argument('--q_cond', type=bool, default=True, help="use q with pose condition")
    parser.add_argument('--uncond_p', type=float, default=0.1, help="probability of uncond classfier free guidance")

    parser.add_argument('--v_pred', type=bool, default=True, help="use v prediction")
    parser.add_argument('--n_particles', type=int, default=1, help="num of particles")
    parser.add_argument('--cube', type=bool, default=True, help="use cube marching box")
    parser.add_argument('--no_textureless', type=bool, default=False, help="no using of textureless")
    parser.add_argument('--no_lambertian', type=bool, default=False, help="no using of lambertian")
    parser.add_argument('--iter512', type=int, default=-1, help="the time to change into 512")
    parser.add_argument('--buffer_size', type=int, default=-1, help="the size of replay buffer")
    parser.add_argument('--sphere_mask', type=bool, default=False, help="bound the sigmas in a sphere of radius [bound]")
    parser.add_argument('--pre_noise', type=bool, default=True, help="Add noise to sigma during training")
    parser.add_argument('--desired_resolution', type=int, default=2048, help="resolution of hashgrid")
    parser.add_argument('--mesh_idx', type=int, default=-1, help="saving this mesh")
    parser.add_argument('--flip_sigma', type=bool, default=False, help="flip the sigmas")
    parser.add_argument('--set_ws', type=str, default='', help="")
    parser.add_argument('--upper_clip', type=float, default=-1, help="make upper sigma zeros")
    parser.add_argument('--side_clip', type=float, default=-1, help="make side sigma zeros")
    parser.add_argument('--dynamic_clip', type=bool, default=False, help="clip the gradient")
    parser.add_argument('--p_normal', type=float, default=0, help="probability to use normal shading")
    parser.add_argument('--p_textureless', type=float, default=0, help="probability to use textureless shading")
    parser.add_argument('--normal', type=bool, default=False, help="optimize with normal")
    parser.add_argument('--upper_clip_m', type=float, default=-100, help="make upper sigma zeros in training")
    parser.add_argument('--complex_bg', type=bool, default=False, help="")
    parser.add_argument('--normal_iters', type=int, default=-1, help="warm up iters using only normals")
    parser.add_argument('--t5_iters', type=int, default=5000, help="change tmax to 500 after this")
    parser.add_argument('--lora', type=bool, default=True, help="Use lora as variational score.")
    parser.add_argument('--sds', type=bool, default=False, help="use SDS instead of VSD")
    parser.add_argument('--finetune', type=bool, default=False, help="only finetune texture")
    parser.add_argument('--note', type=str, default='', help="")
    
    ############# EG3D #####################
    # Required.
    # parser.add_argument('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
    parser.add_argument('--cfg',          help='Base configuration',                                      type=str, required=True)
    parser.add_argument('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
    # parser.add_argument('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=int(min=1), required=True)
    # parser.add_argument('--batch',        help='Total batch size', metavar='INT',                         type=int(min=1), required=True)
    # parser.add_argument('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=float(min=0), required=True)

    # Optional features.
    parser.add_argument('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True)
    # parser.add_argument('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False )
    # parser.add_argument('--aug',          help='Augmentation mode',                                       choices=['noaug', 'ada', 'fixed']), default='noaug' )
    # parser.add_argument('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
    # parser.add_argument('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=int(min=0), default=0 )

    # Misc hyperparameters.
    # parser.add_argument('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=float(min=0, max=1), default=0.2 )
    parser.add_argument('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=float, default=0.6 )
    # parser.add_argument('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=int(min=1))
    parser.add_argument('--cbase',        help='Capacity multiplier', metavar='INT',                      type=int, default=32768 )
    parser.add_argument('--cmax',         help='Max. feature maps', metavar='INT',                        type=int, default=512 )
    # parser.add_argument('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=float(min=0))
    # parser.add_argument('--dlr',          help='D learning rate', metavar='FLOAT',                        type=float(min=0), default=0.002 )
    parser.add_argument('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=int, default=2 )
    parser.add_argument('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=int, default=4 )

    # Misc settings.
    # parser.add_argument('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
    # parser.add_argument('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full' )
    # parser.add_argument('--kimg',         help='Total training duration', metavar='KIMG',                 type=int(min=1), default=25000 )
    # parser.add_argument('--tick',         help='How often to print progress', metavar='KIMG',             type=int(min=1), default=4 )
    # parser.add_argument('--snap',         help='How often to save snapshots', metavar='TICKS',            type=int(min=1), default=50 )
    # parser.add_argument('--seed',         help='Random seed', metavar='INT',                              type=int(min=0), default=0 )
    # parser.add_argument('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False )
    # parser.add_argument('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False )
    # parser.add_argument('--workers',      help='DataLoader worker processes', metavar='INT',              type=int(min=1), default=3 )
    # parser.add_argument('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

    # parser.add_argument('--sr_module',    help='Superresolution module', metavar='STR',  type=str, required=True)
    parser.add_argument('--neural_rendering_resolution_initial', help='Resolution to render at', metavar='INT',  type=int, default=64, required=False)
    parser.add_argument('--neural_rendering_resolution_final', help='Final resolution to render at, if blending', metavar='INT',  type=int, required=False, default=None)
    parser.add_argument('--neural_rendering_resolution_fade_kimg', help='Kimg to blend resolution over', metavar='INT',  type=int, required=False, default=1000 )

    parser.add_argument('--blur_fade_kimg', help='Blur over how many', metavar='INT',  type=int, required=False, default=200)
    parser.add_argument('--gen_pose_cond', help='If true, enable generator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=False)
    parser.add_argument('--c-scale', help='Scale factor for generator pose conditioning.', metavar='FLOAT',  type=float, required=False, default=1)
    parser.add_argument('--c-noise', help='Add noise for generator pose conditioning.', metavar='FLOAT',  type=float, required=False, default=0)
    parser.add_argument('--gpc_reg_prob', help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.', metavar='FLOAT',  type=float, required=False, default=0.5)
    parser.add_argument('--gpc_reg_fade_kimg', help='Length of swapping prob fade', metavar='INT',  type=int, required=False, default=1000)
    parser.add_argument('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=float, required=False, default=0)
    parser.add_argument('--sr_noise_mode', help='Type of noise for superresolution', metavar='STR',  choices=['random', 'none'], required=False, default='none')
    parser.add_argument('--resume_blur', help='Enable to blur even on resume', metavar='BOOL',  type=bool, required=False, default=False)
    parser.add_argument('--sr_num_fp16_res',    help='Number of fp16 layers in superresolution', metavar='INT', type=int, default=4, required=False )
    parser.add_argument('--g_num_fp16_res',    help='Number of fp16 layers in generator', metavar='INT', type=int, default=0, required=False )
    parser.add_argument('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', metavar='INT', type=int, default=4, required=False )
    parser.add_argument('--sr_first_cutoff',    help='First cutoff for AF superresolution', metavar='INT', type=int, default=2, required=False )
    parser.add_argument('--sr_first_stopband',    help='First cutoff for AF superresolution', metavar='FLOAT', type=float, default=2**2.1, required=False )
    parser.add_argument('--style_mixing_prob',    help='Style-mixing regularization probability for training.', metavar='FLOAT', type=float, default=0, required=False )
    parser.add_argument('--sr-module',    help='Superresolution module override', metavar='STR',  type=str, required=False, default=None)
    parser.add_argument('--density_reg',    help='Density regularization strength.', metavar='FLOAT', type=float, default=0.25, required=False )
    parser.add_argument('--density_reg_every',    help='lazy density reg', metavar='int', type=float, default=4, required=False )
    parser.add_argument('--density_reg_p_dist',    help='density regularization strength.', metavar='FLOAT', type=float, default=0.004, required=False )
    parser.add_argument('--reg_type', help='Type of regularization', metavar='STR',  choices=['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation'], required=False, default='l1')
    parser.add_argument('--decoder_lr_mul',    help='decoder learning rate multiplier.', metavar='FLOAT', type=float, default=1, required=False )
    
    parser.add_argument('--use_pretrained',    help='pretrianed model path', metavar='STR',  type=str, required=False, default=None)

    ############# EG3D #####################

    opt = parser.parse_args()
    c = make_c(opt)

    assert opt.p_normal == 0

    if opt.dmtet:
        # parameters for finetuning
        opt.h = 512
        opt.w = 512
        opt.t_range = [0.02, 0.50]
        # opt.fovy_range = [60, 90]
        opt.fovy_range = [30, 60]

    if opt.albedo:
        opt.albedo_iters = opt.iters
        albedostr = "albedo"
    else:
        albedostr = "shading-"+str(opt.albedo_iters)

    opt.val_nz = opt.n_particles

    opt.workspace += str(time.strftime('%Y-%m-%d', time.localtime()))+"-"+str(opt.text).replace(" ", "-")
    if opt.latent == True:
        opt.workspace += "-latent"
        opt.H = 64
        opt.W = 64
    opt.workspace += "-scale-"+str(opt.scale) + "-lr-"+str(opt.lr) 
    opt.workspace += "-" + albedostr+"-le-"+str(opt.lambda_entropy)

    if opt.w != 64:
        assert opt.w == opt.h
        opt.workspace += "-render-" +str(opt.w)
    if opt.cube:
        opt.workspace += "-cube"
    if opt.no_textureless:
        opt.workspace += "-no_textless"
    if opt.suppress_face:
        opt.workspace += "-supface"
    if opt.iter512 != -1:
        opt.workspace += "-iter512-"+str(opt.iter512)
    if opt.buffer_size != -1:
        opt.workspace += "-buffsize-"+str(opt.buffer_size)
    if opt.sphere_mask:
        opt.workspace += "-sphere_mask"
    if opt.bound != 1:
        opt.workspace += "-bound-"+str(opt.bound)
    if opt.sd_version != "1.5":
        opt.workspace += "-sd-"+str(opt.sd_version)        
    if opt.lambda_opacity != 0:
        opt.workspace += "-lo-" + str(opt.lambda_opacity)
    if opt.desired_resolution != 2048:
        opt.workspace += "-g-"+str(opt.desired_resolution)  
    if opt.t5_iters != -1:
        opt.workspace +=  "-"+str(opt.t5_iters)
    if opt.sds:
        opt.workspace += "-sds"
    if opt.normal:
        opt.workspace += "-normal"
    if opt.finetune:
        opt.workspace += "-finetune"
    if opt.num_layers != 1:
        opt.workspace += "-nlayers-" + str(opt.num_layers)
    if opt.density_thresh != 0.1:
        opt.workspace += "-dth-" + str(opt.density_thresh)
    opt.workspace += "-tet-"+str(opt.tet_grid_size)
    if opt.lambda_normal != 0:
        opt.workspace += "-lnorm-" + str(opt.lambda_normal)
    if opt.p_textureless != 0:
        opt.workspace += "-ptext-" + str(opt.p_textureless)
    opt.workspace += opt.note

    if opt.set_ws != "":
        opt.workspace = opt.set_ws

    if opt.seed is not None:
        seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = NeRFNetwork(opt).to(device)
    
    common_kwargs = dict(c_dim=25, img_resolution=IMAGE_RES, img_channels=3)
    model = dnnlib.util.construct_class_by_name(**c.G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    if opt.use_pretrained is not None:
        with open(opt.use_pretrained, 'rb') as f:
            G = pkl.load(f)['G_ema'].train().to(device)
            model.load_state_dict(G.state_dict(), strict=False)
        
        print('using pretrained model')
            
    print('Built G!')

    if opt.dmtet and opt.init_ckpt != '':
        if opt.finetune:
            opt.ckpt = opt.init_ckpt
            model.set_idx()
        else:
            state_dict = torch.load(opt.init_ckpt, map_location=device)
            model.load_state_dict(state_dict['model'], strict=False)
            if opt.cuda_ray:
                model.mean_density = state_dict['mean_density']
            model.set_idx()
            model.init_tet()

    # print(model)

    if opt.test:
        guidance = None # no need to load guidance model at test
        from nerf.sd import StableDiffusion
        guidance = StableDiffusion(device, opt.sd_version, opt.hf_key, opt)

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
        trainer.model.set_idx(opt.mesh_idx)

        if opt.save_mesh:
            trainer.save_mesh()
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.per_iter).dataloader()
            trainer.test(test_loader, name = "test", idx = opt.mesh_idx, shading = "albedo")    
            trainer.test(test_loader, name = "test", idx = opt.mesh_idx, shading = "textureless")   
    else:
        
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.per_iter).dataloader()

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:
            # optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr, finetune = opt.finetune), betas=(0.9, 0.99), eps=1e-15)
            optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            guidance = StableDiffusion(device, opt.sd_version, opt.hf_key, opt)
        elif opt.guidance == 'clip':
            from nerf.clip import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
        trainer.model.set_idx(opt.mesh_idx)
        trainer.test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
        trainer.train_loader512 = NeRFDataset(opt, device=device, type='train', H=512, W=512, size=opt.per_iter).dataloader()

        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.h, W=opt.w, size=opt.val_size).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)
