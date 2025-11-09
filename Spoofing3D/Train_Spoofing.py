# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import time
import warnings
from os import path as osp



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




def Training(is_4D=False):
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
    '''****************************************************************************************************************************************************************************************************************************************'''




    '''定义模型、读取参数文件、锁住参数和BN层****************************************'''
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
    model.train()

    for param in model.parameters():
        param.requires_grad = False
    def fix_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
    model.apply(fix_bn)

    '''定义dataset和dataloader******************************************************'''
    train_dataset = build_dataset(cfg.data.train)
    logger.info(f'trainset contains {len(train_dataset)} frame')

    from mmdet.datasets import build_dataloader as build_mmdet_dataloader
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']
    train_loader = build_mmdet_dataloader(
            train_dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=False,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
    logger.info(f"trainloader contains {len(train_loader)} iter")

    '''定义可学习海报、优化器，并初始化*************************************************'''
    # poster = cv2.imread('Spoofing3D/init_poster.png')
    p_l, p_w = 600,400
    # poster = cv2.resize(poster, (p_l, p_w))
    # poster = torch.from_numpy(poster.astype(np.float32)/255).to('cuda:0') # bgr 0~1
    poster = torch.rand((p_w, p_l, 3), dtype=torch.float32, device='cuda:0')
    learnable_poster = nn.Parameter(poster, requires_grad=True)

    init_lr = 0.01
    decay = 0.5
    max_epoch = 8
    print_iter = 5
    optimizer = torch.optim.Adam([learnable_poster], lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    tv_weight = 10.

    '''训练poster*******************************************************************'''
    device = next(model.parameters()).device
    for epoch_i in range(max_epoch):
        #开始一个epoch的训练
        for iter_i,batch_inputs in enumerate(train_loader):
            batch_inputs = scatter(batch_inputs, [device.index])[0] #放到gpu上
            #put_poster_on_batch_inputs(learnable_poster, batch_inputs, is_bilinear=True)  # 将poster作用到img上
            if not is_4D:
                maskGT_put_poster_on_batch_inputs(learnable_poster, batch_inputs, is_bilinear=True, mask_aug=False)  # 将poster作用到img上
            else: #for bevdet4d
                maskGT_put_poster_on_batch_inputs_4D(learnable_poster, batch_inputs, is_bilinear=True, num_adj=8)  # 将poster作用到img上


            #continue_flag = 0
            #for gt_label in batch_inputs['gt_labels_3d']:
            #    if len(gt_label) == 0: continue_flag = 1
            #if continue_flag: continue

            output = model(return_loss=True, **batch_inputs)#前向传播计算loss
            adv_loss = 0
            for k in output.keys():
                adv_loss += output[k]

            tv_loss = torch.sqrt( 1e-7 + torch.sum((learnable_poster[:p_w-1,:p_l-1,:] - learnable_poster[1:,:p_l-1,:])**2, dim=2) +
                                  torch.sum((learnable_poster[:p_w-1,:p_l-1,:] - learnable_poster[:p_w-1,1:,:])**2, dim=2) ).mean()

            loss = adv_loss + tv_weight * tv_loss



            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([learnable_poster], max_norm=5.)
            optimizer.step()

            learnable_poster.data.copy_( torch.clamp(learnable_poster.data, min=0., max=1. ) )

            if iter_i % print_iter == 0:
                logger.info(f"epoch: {epoch_i}, iter: {iter_i}, adv_loss: {float(adv_loss)}, tv_loss: {float(tv_loss)} and scaled by {tv_weight} ")


        torch.cuda.empty_cache()

        #将学习率×0.95
        if epoch_i%1 == 0:
            init_lr = init_lr*decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            logger.info(f"lr:{init_lr}")

        #保存海报
        torch.save(learnable_poster.data.cpu(), cfg.work_dir+"/poster_%d.pth" % (epoch_i + 1))


if __name__ == '__main__':
    is_BEVDet4D = False

    Training(is_4D=is_BEVDet4D)



