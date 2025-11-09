from __future__ import division
import argparse
import os
import time
import warnings
from os import path as osp
import numpy as np
import cv2

from adv_utils.BEVDet_4D_utils.all_utils import *
import mmcv
import torch
import torch.nn as nn
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import scatter
from mmcv.ops import box_iou_rotated


from Spoofing3D.adv_utils.eval_utils.put_poster_on_batch_inputs import put_poster_on_batch_inputs_eval
from Spoofing3D.adv_utils.parse_args import parse_args
from Spoofing3D.adv_utils.train_utils.maskGT import maskGT_put_poster_on_batch_inputs, \
    maskGT_put_poster_on_batch_inputs_4D
from mmdet3d.apis import init_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

def eval(poster_dir=None,score_thr=0.1, iou_thr=[0.1,0.3,0.5,0.7], center_thr=[0.5,1,2,3], is_4D=False):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.poster_dir is not None:
        poster_dir = args.poster_dir
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

    from mmcv.parallel import collate, scatter
    #model = init_model(args.config, args.checkpoint, device='cuda:0')
    #print(model)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = cfg.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()


    datasets = [build_dataset(cfg.data.val)]
    train_dataset = datasets[0]
    print(f"测试集共包含{len(train_dataset)}帧")

    poster = torch.load(poster_dir).cuda()
    #poster = torch.rand(poster.size()).cuda()
    #poster = torch.from_numpy(cv2.imread('Spoofing3D/init_poster.png').astype(np.float32)/255.).cuda()
    img_poster = (poster.cpu()*255).numpy().astype(np.uint8)
    cv2.imwrite('poster.png',img_poster)

    total_test_frame = 1000
    valid_spoof = 0
    success_spoof_iou = {}
    success_spoof_centerDistant = {}
    for thr in iou_thr:
        success_spoof_iou['iou_%s'%str(thr)] = 0
    for thr in center_thr:
        success_spoof_centerDistant["center_%s"%str(thr)] = 0


    inds = np.random.choice(list(range(len(train_dataset))), total_test_frame, replace=False)
    for jj,ind in enumerate(inds):
        sigle_data_dict = train_dataset[ind]
        #sample_info = train_dataset.data_infos[ind]

        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        ori_gt = data['gt_bboxes_3d'][0][0].tensor

        if not is_4D:
            put_poster_on_batch_inputs_eval(poster, data, mask_aug=False)  # 将poster作用到img上
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8)  # 将poster作用到img上

        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        if len(ori_gt) == len(spoofed_gt): continue
        fake_gt = spoofed_gt[len(ori_gt):,:]
        valid_spoof += len(fake_gt)

        with torch.no_grad(): #前向推理
            result = model(return_loss=False, rescale=True, **data)

        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score > score_thr
        predicted_box = result[0]['pts_bbox']['boxes_3d'].tensor[mask] #(N,9)
        if len(predicted_box) ==0 : continue

        fake_bev = fake_gt[:, [0, 1, 3, 4, 6]]
        pred_bev = predicted_box[:, [0, 1, 3, 4, 6]]
        bev_iou = box_iou_rotated(fake_bev, pred_bev)

        for thr in iou_thr:
            success_spoof_iou['iou_%s' % str(thr)] += (torch.max(bev_iou,dim=1)[0] > thr).sum()


        for i in range(len(fake_bev)):
            min_dis = torch.sqrt(torch.sum((pred_bev[:,:2] - fake_bev[i][:2].unsqueeze(0))**2, dim=1)).min()
            for thr in center_thr:
                if min_dis <thr: success_spoof_centerDistant["center_%s"%str(thr)] += 1

        print("\r", f"{jj}/{total_test_frame}", f"有效伪造目标个数{valid_spoof}", f"iou_thr=0.5，成功伪造{success_spoof_iou['iou_%s' % str(0.5)]}个，成功率为{success_spoof_iou['iou_%s' % str(0.5)]*1./valid_spoof}",
              f"center dis=1，成功伪造{success_spoof_centerDistant['center_%s'%str(1)]}个，成功率为{success_spoof_centerDistant['center_%s'%str(1)]*1./valid_spoof}",end="")

        if jj == 500:
            print(f"有效伪造目标个数{valid_spoof}")
            for thr in iou_thr:
                print(f"BEV iou阈值为{thr}时，成功伪造{success_spoof_iou['iou_%s' % str(thr)]}个，成功率为{success_spoof_iou['iou_%s' % str(thr)] * 1. / valid_spoof}")
            for thr in center_thr:
                print(f"中心点距离阈值为{thr}时，成功伪造{success_spoof_centerDistant['center_%s' % str(thr)]}个，成功率为{success_spoof_centerDistant['center_%s' % str(thr)] * 1. / valid_spoof}")

    print("Finished")
    print(f"有效伪造目标个数{valid_spoof}")

    for thr in iou_thr:
        print(f"BEV iou阈值为{thr}时，成功伪造{success_spoof_iou['iou_%s' % str(thr)]}个，成功率为{success_spoof_iou['iou_%s' % str(thr)] * 1. / valid_spoof}")
    for thr in center_thr:
        print(f"中心点距离阈值为{thr}时，成功伪造{success_spoof_centerDistant['center_%s'%str(thr)]}个，成功率为{success_spoof_centerDistant['center_%s'%str(thr)] * 1. / valid_spoof}")
if __name__ == '__main__':
    is_BEVDet4D = False
    eval( score_thr=0.1, iou_thr=[0.1, 0.3, 0.5, 0.7],
         center_thr=[0.5, 1, 1.5, 2, 3], is_4D=is_BEVDet4D)