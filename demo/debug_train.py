# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp


import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

import numpy as np
import cv2
def mmlabDeNormalize(img):
    from mmcv.image.photometric import imdenormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_bgr = True
    img = img.permute(1,2,0).numpy()
    img = imdenormalize(img, mean, std, to_bgr)
    return img

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs/debug')
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    '''
    #### for modification ####
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    #print(model)

    '''
    datasets = [build_dataset(cfg.data.train)]
    train_dataset = datasets[0]
    print(len(train_dataset))



    ind = 10060
    sigle_data_dict = train_dataset[ind]

    img_inputs = sigle_data_dict['img_inputs']
    imgs = img_inputs[0]
    cam_total = []

    
    for cam_i in range(6):
        cam_col = []
        for adj_i in range(9):
            img = imgs[cam_i*9+adj_i,:,:,:]
            img = mmlabDeNormalize(img)#(3,H,W) rgb
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img) #bgr
            cam_col.append(img)
        cam_total.append(np.hstack(cam_col))
    cam_total = np.vstack(cam_total)
    cv2.imwrite('demo.png', cam_total)


    '''
    img_inputs = sigle_data_dict['img_inputs']
    imgs = img_inputs[0]
    cam_total = []

    for cam_i in range(6):
        img = mmlabDeNormalize(imgs[cam_i])  # (3,H,W) rgb
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr
        cam_total.append(img)

    cam_total = np.vstack(cam_total)
    cv2.imwrite('demo.png', cam_total)
    '''

    '''
    img_inputs = sigle_data_dict['img_inputs']
    imgs = img_inputs[0]
    img = mmlabDeNormalize(imgs[0])#rgb (H,W,3)
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
 
    mask = mmcv.imread('demo/Mud_Mask_selected/mask_2.jpg', 'unchanged')

    def put_mask_on_img(img, mask):
        h, w = img.shape[:2]
        mask = np.rot90(mask)
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = mask / 255
        alpha = np.power(alpha, 3)
        img_with_mask = alpha * img + (1 - alpha) * mask

        return img_with_mask

    img_with_mask = put_mask_on_img(img, mask)


    cv2.imwrite('demo.png', img_with_mask)
    '''





    #print(len(train_dataset.data_infos))
    #print(train_dataset.data_infos[0])
    #print((sigle_data_dict['points'].data)[:, 3].max())
    #print((sigle_data_dict['points'].data)[:, 3].min())
    points = sigle_data_dict['points'].data.numpy()
    bbox = sigle_data_dict['gt_bboxes_3d'].data
    points.tofile('points.bin')
    bbox.corners.numpy().tofile('anno.bin')


    return 0

def main_data():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs/debug')
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    '''
    #### for modification ####
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    #print(model)

    '''
    datasets = [build_dataset(cfg.data.train)]
    train_dataset = datasets[0]
    print(len(train_dataset))



    ind = 55
    sigle_data_dict = train_dataset[ind]

    if 1:
        print(train_dataset.data_infos[ind].keys())
        print(train_dataset.data_infos[ind]['cams']['CAM_FRONT'])
        print('lidar2ego_translation', train_dataset.data_infos[ind]['lidar2ego_translation'])
        print('lidar2ego_rotation', train_dataset.data_infos[ind]['lidar2ego_rotation'])
        print('ego2global_translation', train_dataset.data_infos[ind]['ego2global_translation'])
        print('ego2global_rotation', train_dataset.data_infos[ind]['ego2global_rotation'])

        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

        first_sample = nusc.get('sample', train_dataset.data_infos[ind]['token'])

        first_sample_data_cam_f = nusc.get('sample_data', first_sample['data']['CAM_FRONT'])
        ego_pose = nusc.get('ego_pose', first_sample_data_cam_f['ego_pose_token'])
        calibrated_sensor = nusc.get('calibrated_sensor', first_sample_data_cam_f['calibrated_sensor_token'])

        print('first_sample_data_cam_f', first_sample_data_cam_f)
        print('calibrated_sensor', calibrated_sensor)
        print('ego_pose',ego_pose)


        first_sample_data_lidar = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])
        print('first_sample_data_lidar',first_sample_data_lidar)
        print('calibrated_sensor',nusc.get('calibrated_sensor', first_sample_data_lidar['calibrated_sensor_token']))
        print('ego_pose', nusc.get('ego_pose', first_sample_data_lidar['ego_pose_token']))
        return 0


    if 1:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

        first_sample_token = nusc.scene[0]['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)

        print(first_sample)
        print('')

        first_sample_data_cam_f = nusc.get('sample_data', first_sample['data']['CAM_FRONT'])
        print(first_sample_data_cam_f)

        ego_pose = nusc.get('ego_pose', first_sample_data_cam_f['ego_pose_token'])
        calibrated_sensor = nusc.get('calibrated_sensor', first_sample_data_cam_f['calibrated_sensor_token'])

        print(ego_pose)
        print(calibrated_sensor)

        return 0


    if 1:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

        first_sample_token = nusc.scene[0]['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)

        while True:


            first_sample_data_cam_f = nusc.get('sample_data', first_sample['data']['CAM_FRONT'])
            first_sample_data_cam_fl = nusc.get('sample_data', first_sample['data']['CAM_FRONT_LEFT'])
            first_sample_data_cam_fr = nusc.get('sample_data', first_sample['data']['CAM_FRONT_RIGHT'])
            first_sample_data_cam_b = nusc.get('sample_data', first_sample['data']['CAM_BACK'])
            first_sample_data_cam_bl = nusc.get('sample_data', first_sample['data']['CAM_BACK_LEFT'])
            first_sample_data_cam_br = nusc.get('sample_data', first_sample['data']['CAM_BACK_RIGHT'])
            first_sample_data_lidar = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])

            for cur_sample_data in [first_sample_data_cam_f,first_sample_data_cam_fl,first_sample_data_cam_fr,first_sample_data_cam_b,first_sample_data_cam_bl,first_sample_data_cam_br,first_sample_data_lidar]:
                indicator = []
                while True:
                    indicator.append(cur_sample_data['is_key_frame'])
                    if cur_sample_data['next'] is '': break
                    cur_sample_data = nusc.get('sample_data', cur_sample_data['next'])

                    if cur_sample_data['is_key_frame'] == True: break
                print(indicator)

            print('  ')
            if first_sample['next'] is '': break
            first_sample = nusc.get('sample', first_sample['next'])

        return 0


if __name__ == '__main__':
    #main()
    main_data()
