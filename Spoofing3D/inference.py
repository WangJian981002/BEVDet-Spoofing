from __future__ import division

import time
import warnings
from os import path as osp
import numpy as np
import cv2


import mmcv
import torch

import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint


from Spoofing3D.Train_Spoofing import parse_args
from Spoofing3D.adv_utils.BEVDet_4D_utils.all_utils import put_poster_on_batch_inputs_eval_4D
from Spoofing3D.adv_utils.eval_utils.put_poster_on_batch_inputs import put_poster_on_batch_inputs_eval
from Spoofing3D.adv_utils.visulize_utils.bbox2cam import visulize_3dbox_to_cam

from mmdet3d.apis import init_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

from adv_utils.common_utils import  mmlabDeNormalize

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes
def inference(is_4D=False):
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

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']



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
    print(len(train_dataset))

    poster = torch.load(args.poster_dir).cuda()
    img_poster = (poster.cpu()*255).numpy().astype(np.uint8)
    cv2.imwrite('poster.png',img_poster)



    ind = int(args.id)
    sigle_data_dict = train_dataset[ind]
    sample_info = train_dataset.data_infos[ind]

    #points = sigle_data_dict['points'][0].data.numpy()
    #points.tofile('points.bin')



    from mmcv.parallel import collate, scatter
    device = next(model.parameters()).device
    data = collate([sigle_data_dict], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device.index])[0]

    if not is_4D:
        ori_gt = data['gt_bboxes_3d'][0][0].tensor
        put_poster_on_batch_inputs_eval(poster, data)  # 将poster作用到img上
    else:
        put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8)


    data['gt_bboxes_3d'][0][0].corners.numpy().tofile('anno.bin')
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    #print(result)

    predict_score = result[0]['pts_bbox']['scores_3d']
    mask = predict_score>0.3
    predicted_box = result[0]['pts_bbox']['boxes_3d']
    predicted_box.corners[mask].numpy().tofile('predict_anno.bin')


    if not is_4D:

        vis_imgs = []
        for i in range(6):
            img = data['img_inputs'][0][0][0][i].cpu()
            img = mmlabDeNormalize(img)  # rgb (H,W,3)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],sigle_data_dict['img_inputs'][0][5][i], 256, 704)

            vis_imgs.append(img)

        vis_imgs = np.vstack(vis_imgs)
        cv2.imwrite('inference.png', vis_imgs)
    else:
        cam_total = []
        gt_coner = predicted_box.corners[mask]
        for cam_i in range(6):
            cam_col = []
            for adj_i in range(9):
                img = data['img_inputs'][0][0][0][cam_i * 9 + adj_i, :, :, :].cpu()
                img = mmlabDeNormalize(img)  # (3,H,W) rgb
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr
                cam_col.append(img)
            cam_total.append(np.vstack(cam_col))
        cam_total = np.hstack(cam_total)
        cv2.imwrite('inference.png', cam_total)


if __name__ == '__main__':
    is_BEVDet4D = False
    inference(is_4D=is_BEVDet4D)