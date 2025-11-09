# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp
import shutil


import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model, init_model
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
    parser.add_argument('checkpoint', help='checkpoint file')
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

    ind = 100
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
        cam_total.append(np.vstack(cam_col))
    cam_total = np.hstack(cam_total)
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
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

def inference():
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

    def visulize_3dbox_to_cam(img, gt_coner, sample_info, cam_name, post_rot, post_tran, img_h, img_w):
        #img (H,W,3) BGR
        #gt_corner (N, 8, 3)
        #sample_info dict 信息字典
        #cam_name img是哪个相机
        #post_rot (3,3)由图像增强所带来的旋转矩阵
        #post_tran （3，3）由图像增强所带来的平移矩阵
        #imgh 图像高
        #imgw 图像宽
        from pyquaternion import Quaternion
        lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego[:3, :3] = Quaternion(sample_info['lidar2ego_rotation']).rotation_matrix
        lidar2lidarego[:3, 3] = sample_info['lidar2ego_translation']
        lidar2lidarego = torch.from_numpy(lidar2lidarego)

        lidarego2global = np.eye(4, dtype=np.float32)
        lidarego2global[:3, :3] = Quaternion(sample_info['ego2global_rotation']).rotation_matrix
        lidarego2global[:3, 3] = sample_info['ego2global_translation']
        lidarego2global = torch.from_numpy(lidarego2global)

        cam2camego = np.eye(4, dtype=np.float32)
        cam2camego[:3, :3] = Quaternion(sample_info['cams'][cam_name]['sensor2ego_rotation']).rotation_matrix
        cam2camego[:3, 3] = sample_info['cams'][cam_name]['sensor2ego_translation']
        cam2camego = torch.from_numpy(cam2camego)

        camego2global = np.eye(4, dtype=np.float32)
        camego2global[:3, :3] = Quaternion(sample_info['cams'][cam_name]['ego2global_rotation']).rotation_matrix
        camego2global[:3, 3] = sample_info['cams'][cam_name]['ego2global_translation']
        camego2global = torch.from_numpy(camego2global)

        cam2img = np.eye(4, dtype=np.float32)
        cam2img = torch.from_numpy(cam2img)
        cam2img[:3, :3] = torch.from_numpy(sample_info['cams'][cam_name]['cam_intrinsic'])

        lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(lidarego2global.matmul(lidar2lidarego))
        lidar2img = cam2img.matmul(lidar2cam)

        gt_coner = gt_coner.view(-1,3)
        gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
        gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
        gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)

        gt_coner = gt_coner.view(-1, 8 ,3)


        def is_in_img (p1,p2,img_h,img_w):
            if p1[2] < 1 or p2[2] < 1:
                return False
            flag1 = (p1[0] >= 0) & (p1[0] < img_w) & (p1[1] >= 0) & (p1[1] < img_h) & (p1[2] < 60) & (p1[2] > 1)
            flag2 = (p2[0] >= 0) & (p2[0] < img_w) & (p2[1] >= 0) & (p2[1] < img_h) & (p2[2] < 60) & (p2[2] > 1)
            return flag1 or flag2

        for obj_i in range(len(gt_coner)):
            corner = gt_coner[obj_i] #(8,3)
            connect = [[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]]
            for line_i in connect:
                p1 = corner[line_i[0]]
                p2 = corner[line_i[1]]
                if is_in_img(p1,p2,img_h,img_w):
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), thickness=1)
        #cv2.imwrite('demo.png',img)
        return img
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    model = init_model(args.config, args.checkpoint, device='cuda:0')
    #print(model)






    datasets = [build_dataset(cfg.data.train)]
    train_dataset = datasets[0]
    print(len(train_dataset))

    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=False)

    select_scene_name = 'scene-0165'
    type = 'occ'
    sample_index_list = []

    for i in range(len(train_dataset)):
        sample_i = nusc.get('sample', train_dataset.data_infos[i]['token'])
        scene_i = nusc.get('scene', sample_i['scene_token'])
        if scene_i['name'] == select_scene_name:
            sample_index_list.append(i)
    print(sample_index_list)
    print(len(sample_index_list))
    from mmcv.parallel import collate, scatter

    out_dir = os.path.join('./demo/work_dir', type, select_scene_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(os.path.join(out_dir))
    os.mkdir(os.path.join(out_dir,'points'))
    os.mkdir(os.path.join(out_dir, 'anno'))
    os.mkdir(os.path.join(out_dir, 'predict_anno'))
    os.mkdir(os.path.join(out_dir, 'img'))


    for ind,sample_ind in enumerate(sample_index_list):
        sigle_data_dict = train_dataset[sample_ind]
        sample_info = train_dataset.data_infos[sample_ind]

        # test mode取数据
        points = sigle_data_dict['points'][0].data.numpy()
        bbox = sigle_data_dict['gt_bboxes_3d'][0].data
        points.tofile(os.path.join(out_dir,'points','points_'+str(ind)+'.bin'))
        bbox.corners.numpy().tofile(os.path.join(out_dir,'anno','anno_'+str(ind)+'.bin'))

        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        # print(result)
        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score > 0.2
        predicted_box = result[0]['pts_bbox']['boxes_3d']
        predicted_box.corners[mask].numpy().tofile(os.path.join(out_dir, 'predict_anno', 'predict_anno_' + str(ind) + '.bin'))

        vis_imgs123 = []
        vis_imgs456 = []
        for i in range(3):
            img = sigle_data_dict['img_inputs'][0][0][9 * i]
            img = mmlabDeNormalize(img)  # rgb (H,W,3)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i],
                                        sigle_data_dict['img_inputs'][0][4][i],
                                        sigle_data_dict['img_inputs'][0][5][i], 256, 704)

            vis_imgs123.append(img)

        for i in range(3, 6):
            img = sigle_data_dict['img_inputs'][0][0][9 * i]
            img = mmlabDeNormalize(img)  # rgb (H,W,3)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i],
                                        sigle_data_dict['img_inputs'][0][4][i],
                                        sigle_data_dict['img_inputs'][0][5][i], 256, 704)

            vis_imgs456.append(img)

        vis_imgs123 = np.hstack(vis_imgs123)
        vis_imgs456 = np.hstack(vis_imgs456)
        vis_imgs = np.vstack([vis_imgs123, vis_imgs456])

        cv2.imwrite(os.path.join(out_dir, 'img', 'img_' + str(ind) + '.png'), vis_imgs)


















    '''
    ind = 0
    sigle_data_dict = train_dataset[ind]
    sample_info = train_dataset.data_infos[ind]

    #test mode取数据
    points = sigle_data_dict['points'][0].data.numpy()
    bbox = sigle_data_dict['gt_bboxes_3d'][0].data
    points.tofile('points.bin')
    bbox.corners.numpy().tofile('anno.bin')


    from mmcv.parallel import collate, scatter
    device = next(model.parameters()).device
    data = collate([sigle_data_dict], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device.index])[0]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    #print(result)
    predict_score = result[0]['pts_bbox']['scores_3d']
    mask = predict_score>0.2
    predicted_box = result[0]['pts_bbox']['boxes_3d']
    predicted_box.corners[mask].numpy().tofile('predict_anno.bin')



    vis_imgs123 = []
    vis_imgs456 = []
    for i in range(3):
        img = sigle_data_dict['img_inputs'][0][0][9*i]
        img = mmlabDeNormalize(img)  # rgb (H,W,3)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
        img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],
                                    sigle_data_dict['img_inputs'][0][5][i], 256, 704)

        vis_imgs123.append(img)

    for i in range(3,6):
        img = sigle_data_dict['img_inputs'][0][0][9*i]
        img = mmlabDeNormalize(img)  # rgb (H,W,3)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
        img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],
                                    sigle_data_dict['img_inputs'][0][5][i], 256, 704)

        vis_imgs456.append(img)

    vis_imgs123 = np.hstack(vis_imgs123)
    vis_imgs456 = np.hstack(vis_imgs456)
    vis_imgs = np.vstack([vis_imgs123, vis_imgs456])

    cv2.imwrite('demo.png', vis_imgs)
    '''



if __name__ == '__main__':
    inference()
