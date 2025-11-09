# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp
import numpy as np
import cv2
import random

import mmcv
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import collate, scatter
from mmcv.ops import box_iou_rotated

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model, init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from matplotlib.path import Path
from adv_utils.common_utils import rotate_points_along_z
from adv_utils.dcgan import DCGAN_G_CustomAspectRatio, DCGAN_D_CustomAspectRatio, weights_init, SceneSet
import lpips

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--resume_netD', default='', type=str, help='resume from this discriminator checkpoint')
    parser.add_argument('--resume_netG', default='', type=str, help='resume from this generator checkpoint')
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

def mmlabDeNormalize(img):
    from mmcv.image.photometric import imdenormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_bgr = True
    img = img.permute(1,2,0).numpy()
    img = imdenormalize(img, mean, std, to_bgr)
    return img

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

def visulize_3dbox_to_cam_multiFrame(img, gt_coner,cam2lidar_r,cam2lidar_t,intrin, post_rot, post_tran, img_h, img_w):
    #img (H,W,3) BGR
    #gt_corner (N, 8, 3)
    #sample_info dict 信息字典
    #cam_name img是哪个相机
    #post_rot (3,3)由图像增强所带来的旋转矩阵
    #post_tran （3，3）由图像增强所带来的平移矩阵
    #imgh 图像高
    #imgw 图像宽

    cam2lidar = torch.eye(4)
    cam2lidar[:3,:3] = cam2lidar_r
    cam2lidar[:3,3] = cam2lidar_t
    lidar2cam = torch.inverse(cam2lidar)

    cam2img = torch.eye(4)
    cam2img[:3,:3] = intrin

    lidar2img = cam2img.matmul(lidar2cam)


    gt_coner = gt_coner.view(-1,3)
    gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
    gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)

    gt_coner = gt_coner.view(-1, 8 ,3)

    def is_in_img(p1, p2, img_h, img_w):
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
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), thickness=1)#(64,128,255)
    #cv2.imwrite('demo.png',img)
    return img

def draw_box_from_batch(img,batch_inputs,frame_i,cam_i,img_h=256,img_w=704):
    cam2lidar = torch.eye(4)
    cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
    cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
    lidar2cam = torch.inverse(cam2lidar)

    cam2img = torch.eye(4)
    cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
    lidar2img = cam2img.matmul(lidar2cam)

    gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners
    post_rot = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu()
    post_tran = batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()

    gt_coner = gt_coner.view(-1, 3)
    gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
    gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
    gt_coner = gt_coner.view(-1, 8, 3)

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


#将图像中的gt bbox mask掉
def maskGT_put_poster_on_batch_inputs(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], is_bilinear=False, mask_aug=False):
    #leaning_poster (m,3,200,300)
    use_poster_idx=0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0].size(0)
    '''mask gt bbox'''
    for frame_i in range(batchsize):
        for cam in cam_names:
            cam_i = camname_idx[cam]
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), \
                                  batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()

            gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners #(G,8,3)
            gt_coner = gt_coner.view(-1, 3)
            gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
            gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            gt_coner = gt_coner.view(-1, 8, 3)

            for gt_i in range(len(gt_coner)):
                corner_3d = gt_coner[gt_i,:,:] #(8,3) [u,v,z]
                #if corner_3d[:,2].min() <=1 : continue
                if (corner_3d[:,2]>1).sum() <= 2 : continue
                corner_3d = corner_3d[corner_3d[:,2] > 1]

                xmin, ymin = corner_3d[:,0].min(), corner_3d[:,1].min()
                xmax, ymax = corner_3d[:,0].max(), corner_3d[:,1].max()

                if xmin > img_w-1 or ymin > img_h - 1 or xmax <= 0 or ymax <= 0 : continue
                xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(img_w-1, int(xmax)), min(img_h-1, int(ymax))
                batch_inputs['img_inputs'][0][frame_i,cam_i,:,ymin:ymax,xmin:xmax] = ((torch.Tensor([[0.5,0.5,0.5]])- torch.from_numpy(mean)) / torch.from_numpy(std)).cuda().permute(1,0).unsqueeze(-1)


    '''put poster on image'''
    for frame_i in range(batchsize):
        num_obj_in_frame = len(batch_inputs['gt_bboxes_3d'][frame_i].tensor)

        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1. : continue #防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            if mask_aug:
                if np.random.random() < 0.4:
                    leaning_poster_clone = leaning_poster
                else:
                    leaning_poster_clone = leaning_poster.clone()
                    for p_i in range(leaning_poster.size(0)):
                        bound_pixel = 50
                        start_w = np.random.choice(poster_w-bound_pixel)
                        start_l = np.random.choice(poster_l-bound_pixel)
                        end_w = min(start_w+50, poster_w)
                        end_l = min(start_l+50, poster_l)
                        leaning_poster_clone[p_i,:,start_w:end_w, start_l:end_l] = 0

            else:
                leaning_poster_clone = leaning_poster


            if is_bilinear:
                index_l = torch.clip((lidar_inner_points[:, 0] / l)*2-1, min=-1, max=1)
                index_w = torch.clip(((w - lidar_inner_points[:, 1]) / w)*2-1, min=-1, max=1)
                grid = torch.cat([index_l.unsqueeze(-1), index_w.unsqueeze(-1)], dim=1).unsqueeze(0).unsqueeze(0) #(1,1,Nin,2)
                selected_color = torch.nn.functional.grid_sample(leaning_poster_clone[use_poster_idx].unsqueeze(0), grid.cuda(), mode='bilinear', align_corners=True) #(1,3,1,Nin)
                selected_color = selected_color.squeeze().permute(1,0)

            else :
                index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                selected_color = leaning_poster_clone[use_poster_idx, :, index_w, index_l].T #(Nin, 3) bgr 0~1 gpu
            use_poster_idx+=1

            contrast = round(random.uniform(0.8, 1.0), 10)
            brightness = round(random.uniform(-0.15, 0.1), 10)
            selected_color = selected_color * contrast + brightness
            selected_color[selected_color > 1] = 1
            selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][frame_i]
            batch_inputs['gt_labels_3d'][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])
        #只保留spoofer的gt信息
        batch_inputs['gt_bboxes_3d'][frame_i].tensor = batch_inputs['gt_bboxes_3d'][frame_i].tensor[num_obj_in_frame:,:]
        batch_inputs['gt_labels_3d'][frame_i] = batch_inputs['gt_labels_3d'][frame_i][num_obj_in_frame:].long()

#测试时候放到scatter后的batch inputs里
def put_poster_on_batch_inputs_eval(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], mask_aug=False,use_next_poster=False):
    use_poster_idx = 0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4, 2)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0][0].size(0)
    for frame_i in range(batchsize):
        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][0][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][0][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][0][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

            if mask_aug:
                leaning_poster_clone = leaning_poster.clone()
                for p_i in range(len(leaning_poster_clone)):
                    bound_pixel = 50
                    start_w = np.random.choice(poster_w - bound_pixel)
                    start_l = np.random.choice(poster_l - bound_pixel)
                    end_w = min(start_w + 50, poster_w)
                    end_l = min(start_l + 50, poster_l)
                    leaning_poster_clone[p_i,:,start_w:end_w, start_l:end_l] = 0
            else:
                leaning_poster_clone = leaning_poster.clone()


            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
            index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
            selected_color = leaning_poster_clone[use_poster_idx,:,index_w, index_l].T #(Nin, 3) bgr 0~1 gpu
            if use_next_poster:
                use_poster_idx += 1

            #contrast = round(random.uniform(0.7, 1.0), 10)
            #brightness = round(random.uniform(-0.3, 0.2), 10)
            #selected_color = selected_color * contrast + brightness
            #selected_color[selected_color > 1] = 1
            #selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][0][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][0][frame_i]
            batch_inputs['gt_labels_3d'][0][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])

#第二阶段训练时，可指定放置的location
def maskGT_put_poster_on_batch_inputs_v2(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], mask_aug=False,location_dict=None,use_next_poster=True):
    use_poster_idx = 0
    if location_dict is None:
        is_sampling_loaction = True
        location_dict = []
    else:
        is_sampling_loaction = False


    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0].size(0)
    '''mask gt bbox'''
    for frame_i in range(batchsize):
        for cam in cam_names:
            cam_i = camname_idx[cam]
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), \
                                  batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()

            gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners  # (G,8,3)
            gt_coner = gt_coner.view(-1, 3)
            gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
            gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            gt_coner = gt_coner.view(-1, 8, 3)

            for gt_i in range(len(gt_coner)):
                corner_3d = gt_coner[gt_i, :, :]  # (8,3) [u,v,z]
                # if corner_3d[:,2].min() <=1 : continue
                if (corner_3d[:, 2] > 1).sum() <= 2: continue
                corner_3d = corner_3d[corner_3d[:, 2] > 1]

                xmin, ymin = corner_3d[:, 0].min(), corner_3d[:, 1].min()
                xmax, ymax = corner_3d[:, 0].max(), corner_3d[:, 1].max()

                if xmin > img_w - 1 or ymin > img_h - 1 or xmax <= 0 or ymax <= 0: continue
                xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(img_w - 1, int(xmax)), min(img_h - 1,
                                                                                                              int(ymax))
                batch_inputs['img_inputs'][0][frame_i, cam_i, :, ymin:ymax, xmin:xmax] = (
                            (torch.Tensor([[0.5, 0.5, 0.5]]) - torch.from_numpy(mean)) / torch.from_numpy(
                        std)).cuda().permute(1, 0).unsqueeze(-1)

    for frame_i in range(batchsize):
        num_obj_in_frame = len(batch_inputs['gt_bboxes_3d'][frame_i].tensor)

        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            if is_sampling_loaction:
                search_flag = 0
                for _ in range(max_search_num):
                    r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                    an = (2 * np.random.rand() - 1) * (5*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                    yaw = (2 * np.random.rand() - 1) * (0 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                    cx = r * np.cos(an)
                    cy = r * np.sin(an)

                    fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                    fake_box = torch.from_numpy(fake_box)

                    bev_iou = box_iou_rotated(fake_box, gt_bev)
                    if len(gt_bev) == 0:
                        break
                    if bev_iou.max() == 0:
                        break
                    search_flag += 1
                if search_flag == max_search_num:
                    location_dict.append(dict(valid=False))
                    continue
                else:
                    location_dict.append(dict(valid=True, r=r, an=an, yaw=yaw))
            else:
                if location_dict[0]['valid']:
                    r, an, yaw = location_dict[0]['r'], location_dict[0]['an'], location_dict[0]['yaw']
                    cx = r * np.cos(an)
                    cy = r * np.sin(an)

                    fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                    fake_box = torch.from_numpy(fake_box)

                    location_dict.pop(0)
                else:
                    location_dict.pop(0)
                    continue


            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

            if mask_aug:
                leaning_poster_clone = leaning_poster.clone()
                for p_i in range(len(leaning_poster_clone)):
                    bound_pixel = 50
                    start_w = np.random.choice(poster_w - bound_pixel)
                    start_l = np.random.choice(poster_l - bound_pixel)
                    end_w = min(start_w + 50, poster_w)
                    end_l = min(start_l + 50, poster_l)
                    leaning_poster_clone[p_i,:,start_w:end_w, start_l:end_l] = 0
            else:
                leaning_poster_clone = leaning_poster.clone()


            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
            index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
            selected_color = leaning_poster_clone[use_poster_idx,:,index_w, index_l].T #(Nin, 3) bgr 0~1 gpu
            if use_next_poster:
                use_poster_idx += 1

            #contrast = round(random.uniform(0.7, 1.0), 10)
            #brightness = round(random.uniform(-0.3, 0.2), 10)
            #selected_color = selected_color * contrast + brightness
            #selected_color[selected_color > 1] = 1
            #selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][frame_i]
            batch_inputs['gt_labels_3d'][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])

        # 只保留spoofer的gt信息
        batch_inputs['gt_bboxes_3d'][frame_i].tensor = batch_inputs['gt_bboxes_3d'][frame_i].tensor[num_obj_in_frame:, :]
        batch_inputs['gt_labels_3d'][frame_i] = batch_inputs['gt_labels_3d'][frame_i][num_obj_in_frame:].long()

    return location_dict


#测试时候放到scatter后的batch inputs里 跨相机
def put_poster_on_batch_inputs_eval_cross_cam(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK']):
    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -160, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -20}
    sample_range = (6, 12)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (3.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[:2]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0][0].size(0)
    for frame_i in range(batchsize):
        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][0][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (48*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (10*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][0][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][0][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)


            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
            index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
            selected_color = leaning_poster[index_w, index_l, :] #(Nin, 3) bgr 0~1 gpu

            #contrast = round(random.uniform(0.7, 1.0), 10)
            #brightness = round(random.uniform(-0.3, 0.2), 10)
            #selected_color = selected_color * contrast + brightness
            #selected_color[selected_color > 1] = 1
            #selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][0][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][0][frame_i]
            batch_inputs['gt_labels_3d'][0][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])

            for ccam_i, ccam in enumerate(cam_names):
                if ccam == spoofcam: continue
                # 求解对应于图像中的四个角点*************************************************'''
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][ccam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][ccam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)
                cam2img = torch.eye(4)
                cam2img[:3, :3] = batch_inputs['img_inputs'][0][3][frame_i][ccam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
                post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][ccam_i].cpu(), \
                                      batch_inputs['img_inputs'][0][5][frame_i][ccam_i].cpu()
                img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                if (img_corner[:, 2] > 0).sum() < 4: continue
                img_corner = img_corner[:, :2]  # (4,2)
                # 求解图像区域内的所有像素点坐标***************************************************************************************'''
                path = Path(img_corner.numpy())
                x, y = np.mgrid[:img_w, :img_h]
                points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
                mask = path.contains_points(points)
                path_points = points[np.where(mask)]  # (Nin,2) [x,y]
                img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
                if len(img_inner_points) == 0: continue
                '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
                img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(
                    torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
                R = torch.inverse(lidar2img[:3, :3].T)
                T = lidar2img[:3, 3]

                fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
                fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
                C = fz / fm  # (Nin)
                img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                                  (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                                  C.unsqueeze(-1)], dim=1)
                lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)
                '''找到每个3D点在poster上的颜色索引,并put到原图上'''
                lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
                lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0),-1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
                lidar_inner_points[:, 0] += l / 2.
                lidar_inner_points[:, 1] += w / 2.

                index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                selected_color = leaning_poster[index_w, index_l, :]  # (Nin, 3) bgr 0~1 gpu

                selected_color = (selected_color - torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()  # (Nin, 3) 归一化
                batch_inputs['img_inputs'][0][0][frame_i, ccam_i, :, img_inner_points[:, 1],img_inner_points[:, 0]] = selected_color.T


#for BEVDet4D
def maskGT_put_poster_on_batch_inputs_4D(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], is_bilinear=False, num_adj=8,use_next_poster=True):
    #leaning_poster (m,3,200,300)
    use_poster_idx=0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0].size(0)
    '''mask gt bbox'''
    for frame_i in range(batchsize):
        for time_i in range(num_adj+1):
            for cam in cam_names:
                cam_i = camname_idx[cam]
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][6*time_i+cam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][6*time_i+cam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)

                cam2img = torch.eye(4)
                cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][6*time_i+cam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][6*time_i+cam_i].cpu(), \
                                      batch_inputs['img_inputs'][5][frame_i][6*time_i+cam_i].cpu()

                gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners #(G,8,3)
                gt_coner = gt_coner.view(-1, 3)
                gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
                gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                gt_coner = gt_coner.view(-1, 8, 3)

                for gt_i in range(len(gt_coner)):
                    corner_3d = gt_coner[gt_i,:,:] #(8,3) [u,v,z]
                    #if corner_3d[:,2].min() <=1 : continue
                    if (corner_3d[:,2]>1).sum() <= 2 : continue
                    corner_3d = corner_3d[corner_3d[:,2] > 1]

                    xmin, ymin = corner_3d[:,0].min(), corner_3d[:,1].min()
                    xmax, ymax = corner_3d[:,0].max(), corner_3d[:,1].max()

                    if xmin > img_w-1 or ymin > img_h - 1 or xmax <= 0 or ymax <= 0 : continue
                    xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(img_w-1, int(xmax)), min(img_h-1, int(ymax))
                    batch_inputs['img_inputs'][0][frame_i, (num_adj+1)*cam_i+time_i, :,ymin:ymax,xmin:xmax] = ((torch.Tensor([[0.5,0.5,0.5]])- torch.from_numpy(mean)) / torch.from_numpy(std)).cuda().permute(1,0).unsqueeze(-1)


    '''put poster on image'''
    for frame_i in range(batchsize):
        num_obj_in_frame = len(batch_inputs['gt_bboxes_3d'][frame_i].tensor)

        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1. : continue #防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)

            for time_i in range(num_adj+1):
                '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][6*time_i+cam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][6*time_i+cam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)

                cam2img = torch.eye(4)
                cam2img[:3,:3] = batch_inputs['img_inputs'][3][frame_i][6*time_i+cam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                if (img_corner[:, 2] < 1).sum(): continue
                img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
                post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][6*time_i+cam_i].cpu(), batch_inputs['img_inputs'][5][frame_i][6*time_i+cam_i].cpu()
                img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                img_corner = img_corner[:, :2]  # (4,2)

                '''求解图像区域内的所有像素点坐标***************************************************************************************'''
                path = Path(img_corner.numpy())
                x, y = np.mgrid[:img_w, :img_h]
                points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
                mask = path.contains_points(points)
                path_points = points[np.where(mask)]  # (Nin,2) [x,y]
                img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
                if time_i == 0 and len(img_inner_points) <= 200: break
                if time_i != 0 and len(img_inner_points) <= 10 : continue

                '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
                img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
                R = torch.inverse(lidar2img[:3, :3].T)
                T = lidar2img[:3, 3]

                fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
                fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
                C = fz / fm  # (Nin)
                img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                                  (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                                  C.unsqueeze(-1)], dim=1)
                lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

                '''找到每个3D点在poster上的颜色索引,并put到原图上'''
                lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
                lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
                lidar_inner_points[:, 0] += l / 2.
                lidar_inner_points[:, 1] += w / 2.

                if is_bilinear:
                    index_l = torch.clip((lidar_inner_points[:, 0] / l)*2-1, min=-1, max=1)
                    index_w = torch.clip(((w - lidar_inner_points[:, 1]) / w)*2-1, min=-1, max=1)
                    grid = torch.cat([index_l.unsqueeze(-1), index_w.unsqueeze(-1)], dim=1).unsqueeze(0).unsqueeze(0) #(1,1,Nin,2)
                    selected_color = torch.nn.functional.grid_sample(leaning_poster[use_poster_idx].unsqueeze(0), grid.cuda(), mode='bilinear', align_corners=True) #(1,3,1,Nin)
                    selected_color = selected_color.squeeze().permute(1,0)

                else :
                    index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                    index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                    selected_color = leaning_poster[use_poster_idx, :, index_w, index_l].T #(Nin, 3) bgr 0~1 gpu

                contrast = round(random.uniform(0.8, 1.0), 10)
                brightness = round(random.uniform(-0.15, 0.1), 10)
                selected_color = selected_color * contrast + brightness
                selected_color[selected_color > 1] = 1
                selected_color[selected_color < 0] = 0

                selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
                batch_inputs['img_inputs'][0][frame_i, (num_adj+1)*cam_i+time_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

                if time_i == 0:
                    batch_inputs['gt_bboxes_3d'][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
                    gt_label = batch_inputs['gt_labels_3d'][frame_i]
                    batch_inputs['gt_labels_3d'][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])
            if use_next_poster:
                use_poster_idx += 1
        #只保留spoofer的gt信息
        batch_inputs['gt_bboxes_3d'][frame_i].tensor = batch_inputs['gt_bboxes_3d'][frame_i].tensor[num_obj_in_frame:,:]
        batch_inputs['gt_labels_3d'][frame_i] = batch_inputs['gt_labels_3d'][frame_i][num_obj_in_frame:].long()

def put_poster_on_batch_inputs_eval_4D(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], num_adj=8,use_next_poster=True, only_apply_cur_frame=False):
    use_poster_idx = 0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0][0].size(0)
    for frame_i in range(batchsize):
        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][0][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            for time_i in range(num_adj + 1):
                '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][6*time_i+cam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][6*time_i+cam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)

                cam2img = torch.eye(4)
                cam2img[:3,:3] = batch_inputs['img_inputs'][0][3][frame_i][6*time_i+cam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                if (img_corner[:, 2] < 1).sum(): continue
                img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
                post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][6*time_i+cam_i].cpu(), batch_inputs['img_inputs'][0][5][frame_i][6*time_i+cam_i].cpu()
                img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                img_corner = img_corner[:, :2]  # (4,2)

                '''求解图像区域内的所有像素点坐标***************************************************************************************'''
                path = Path(img_corner.numpy())
                x, y = np.mgrid[:img_w, :img_h]
                points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
                mask = path.contains_points(points)
                path_points = points[np.where(mask)]  # (Nin,2) [x,y]
                img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
                if time_i == 0 and len(img_inner_points) <= 200: break
                if time_i != 0 and len(img_inner_points) <= 10: continue

                '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
                img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
                R = torch.inverse(lidar2img[:3, :3].T)
                T = lidar2img[:3, 3]

                fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
                fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
                C = fz / fm  # (Nin)
                img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                                  (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                                  C.unsqueeze(-1)], dim=1)
                lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

                '''找到每个3D点在poster上的颜色索引,并put到原图上'''
                lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
                lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
                lidar_inner_points[:, 0] += l / 2.
                lidar_inner_points[:, 1] += w / 2.

                index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                selected_color = leaning_poster[use_poster_idx,:,index_w, index_l].T #(Nin, 3) bgr 0~1 gpu

                #contrast = round(random.uniform(0.7, 1.0), 10)
                #brightness = round(random.uniform(-0.3, 0.2), 10)
                #selected_color = selected_color * contrast + brightness
                #selected_color[selected_color > 1] = 1
                #selected_color[selected_color < 0] = 0

                selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
                batch_inputs['img_inputs'][0][0][frame_i, (num_adj+1)*cam_i+time_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

                if time_i == 0:
                    batch_inputs['gt_bboxes_3d'][0][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
                    gt_label = batch_inputs['gt_labels_3d'][0][frame_i]
                    batch_inputs['gt_labels_3d'][0][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])
                if only_apply_cur_frame and (time_i>2):#仅将补丁渲染至t=0的帧
                    break


            if use_next_poster:
                use_poster_idx += 1



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

    def act_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True


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

    '''超参数'''
    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    batch_size_D = 32
    scene_set_dir = 'data/Background_scene/RoadSnip-ori'

    lrD, lrG = 0.0001, 0.0001
    beta1 = 0.5
    max_epoch = 20
    G_iter_num = 4
    D_iter_num = 0#10
    clamp_lower, clamp_upper = -0.01, 0.01# 限制discriminator参数的范围，参考WGAN

    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]
    adv_weight = 0.02
    tv_weight = 0#0.1
    D_weight = 1.
    print_iter = 1


    '''定义生成器，判别器、和场景数据集*************************************************'''
    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.apply(weights_init)
    if args.resume_netG != '':
        netG.load_state_dict(torch.load(args.resume_netG))
        logger.info(f"load G ckpt from: {str(args.resume_netG)} ")
    netG.to('cuda:0')
    netD = DCGAN_D_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ndf=64, n_extra_layers=0)
    netD.apply(weights_init)
    if args.resume_netD != '':
        netD.load_state_dict(torch.load(args.resume_netD))
        logger.info(f"load D ckpt from: {str(args.resume_netD)} ")
    netD.to('cuda:0')

    Scenedataset = SceneSet(scene_set_dir, imgsize=[poster_size_inside_net,poster_size_inside_net])
    Scenedataloader = DataLoader(Scenedataset, batch_size=batch_size_D, shuffle=True)
    logger.info(f"scene set contains {len(Scenedataset)} imgs")
    Scene_iter = iter(Scenedataloader)
    Scene_iter_count = 1



    '''定义优化器，并初始化*************************************************'''
    if args.adam:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
    else:
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=lrD)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lrG)
    one = torch.FloatTensor([1]).to('cuda:0')
    mone = (one * -1).to('cuda:0')

    errD_real, errD_fake = 0, 0
    '''训练poster*******************************************************************'''
    device = next(model.parameters()).device
    for epoch_i in range(max_epoch):
        #开始一个epoch的训练
        for iter_i,batch_inputs in enumerate(train_loader):
            ############################
            # (1) Update D network
            ###########################
            if iter_i%G_iter_num == 0 :
                for p in netD.parameters():
                    p.requires_grad = True
                netD.apply(act_bn)
                for p in netG.parameters():
                    p.requires_grad = False
                netG.apply(fix_bn)

                for _ in range(D_iter_num):
                    for p in netD.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)  # 限制discriminator参数的范围，参考WGAN

                    if Scene_iter_count<len(Scenedataloader):
                        realposter = Scene_iter.next()
                        Scene_iter_count += 1
                    else:
                        Scene_iter = iter(Scenedataloader)
                        realposter = Scene_iter.next()
                        Scene_iter_count = 2
                    bs_cur = realposter.size(0)
                    # realposter 0~1
                    netD.zero_grad()
                    errD_real = netD(realposter.to('cuda:0'))
                    errD_real.backward(one)

                    noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
                    fakeposter = (netG(noise)+1)/2.0 #(0~1)
                    errD_fake = netD(fakeposter)
                    errD_fake.backward(mone)

                    optimizerD.step()
                    #logger.info(f"errD_real: {float(errD_real)}, errD_fake: {float(errD_fake)}")

            ############################
            # (2) Update G network
            ###########################
            for p in netG.parameters():
                p.requires_grad = True
            netG.apply(act_bn)
            for p in netD.parameters():
                p.requires_grad = False
            netD.apply(fix_bn)

            bs_cur = batch_inputs['img_inputs'][0].size(0)*2 #每帧最多放放在两个相机
            noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
            GAN_out = (netG(noise)+1)/2.
            spoofing_poster = F.interpolate(GAN_out, size=poster_digital_size, mode="bilinear", align_corners=True)#(m,3,200,300) 0~1 bgr



            batch_inputs = scatter(batch_inputs, [device.index])[0] #放到gpu上
            if not is_4D:
                maskGT_put_poster_on_batch_inputs(spoofing_poster, batch_inputs, is_bilinear=True, mask_aug=False)  # 将poster作用到img上
            else: #for bevdet4d
                maskGT_put_poster_on_batch_inputs_4D(spoofing_poster, batch_inputs, is_bilinear=True, num_adj=8,use_next_poster=True)  # 将poster作用到img上

            '''
            vis_imgs = []
            frame_i = 2
            for i in range(6):
                img =batch_inputs['img_inputs'][0][frame_i][i].detach().cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = draw_box_from_batch(img,batch_inputs,frame_i,i)
                vis_imgs.append(img)
    
            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite('demo.png', vis_imgs)
            assert False
            '''


            '''
            cam_total = []
            frame_i = 6
            gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners
            for cam_i in range(6):
                cam_col = []
                for adj_i in range(9):
                    img = batch_inputs['img_inputs'][0][frame_i][cam_i * 9 + adj_i, :, :, :].detach().cpu()
                    img = mmlabDeNormalize(img)  # (3,H,W) rgb
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr
                    img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                           batch_inputs['img_inputs'][1][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][2][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][3][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][4][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][5][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           256, 704)
                    cam_col.append(img)
                cam_total.append(np.hstack(cam_col))
            cam_total = np.vstack(cam_total)
            cv2.imwrite('demo.png', cam_total)
            assert False
            '''

            #continue_flag = 0
            #for gt_label in batch_inputs['gt_labels_3d']:
            #    if len(gt_label) == 0: continue_flag = 1
            #if continue_flag: continue

            output = model(return_loss=True, **batch_inputs)#前向传播计算loss
            adv_loss = 0
            for k in output.keys():
                if 'heatmap' in k:
                    adv_loss += output[k]
                else:
                    adv_loss += 1*output[k] #为了节省显存
                #adv_loss += output[k]

            tv_loss = torch.sqrt( 1e-7 + torch.sum((spoofing_poster[:,:,:p_w-1,:p_l-1] - spoofing_poster[:,:,1:,:p_l-1])**2, dim=1) +
                                  torch.sum((spoofing_poster[:,:,:p_w-1,:p_l-1] - spoofing_poster[:,:,:p_w-1,1:])**2, dim=1) ).mean()

            D_loss = netD(GAN_out)

            #loss = adv_weight * adv_loss + tv_weight * tv_loss + D_weight* D_loss
            loss = adv_weight * adv_loss



            loss.backward()
            if iter_i % G_iter_num == 0:
                optimizerG.step()
                optimizerG.zero_grad()


            if iter_i % G_iter_num == 0:
                logger.info(f"epoch: {epoch_i}, iter: {iter_i}, adv_loss: {float(adv_loss)}, D_loss: {float(D_loss)}, tv_loss: {float(tv_loss)}, errD_real: {float(errD_real)}, errD_fake: {float(errD_fake)}")
            if iter_i % 200 == 0:
                vis_img = torch.flip(GAN_out.cpu().detach(), [1]) #(m,3,256,256) 0~1 rgb
                torchvision.utils.save_image(vis_img, os.path.join(cfg.work_dir,f"e{epoch_i + 1}i{iter_i}.jpg"))



            #torch.cuda.empty_cache()


        #保存海报
        torch.save(netG.state_dict(), os.path.join(cfg.work_dir, f'netG_epoch{epoch_i + 1}.pth'))
        torch.save(netD.state_dict(), os.path.join(cfg.work_dir, f'netD_epoch{epoch_i + 1}.pth'))


def Training_pure_GAN():
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


    def fix_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False

    def act_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True


    '''超参数'''
    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    batch_size_D = 32

    lrD, lrG = 0.0001, 0.00003
    beta1 = 0.5
    max_epoch = 500
    G_iter_num = 1
    D_iter_num = 20
    clamp_lower, clamp_upper = -0.01, 0.01  # 限制discriminator参数的范围，参考WGAN


    print_iter = 1

    '''定义生成器，判别器、和场景数据集*************************************************'''
    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.apply(weights_init)
    netG.to('cuda:0')
    netD = DCGAN_D_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ndf=64, n_extra_layers=0)
    netD.apply(weights_init)
    netD.to('cuda:0')

    Scenedataset = SceneSet('data/Background_scene/RoadSnip', imgsize=[poster_size_inside_net, poster_size_inside_net])
    Scenedataloader = DataLoader(Scenedataset, batch_size=batch_size_D, shuffle=True)
    logger.info(f"scene set contains {len(Scenedataset)} imgs")
    Scene_iter = iter(Scenedataloader)
    Scene_iter_count = 1

    '''定义优化器，并初始化*************************************************'''
    if args.adam:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
    else:
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=lrD)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lrG)
    one = torch.FloatTensor([1]).to('cuda:0')
    mone = (one * -1).to('cuda:0')

    '''训练poster*******************************************************************'''
    for epoch_i in range(max_epoch):
        # 开始一个epoch的训练
        for iter_i in range(1000):
            if iter_i % G_iter_num == 0:
                ############################
                # (1) Update D network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = True
                netD.apply(act_bn)
                for p in netG.parameters():
                    p.requires_grad = False
                netG.apply(fix_bn)

                for _ in range(D_iter_num):
                    for p in netD.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)  # 限制discriminator参数的范围，参考WGAN

                    if Scene_iter_count < len(Scenedataloader):
                        realposter = Scene_iter.next()
                        Scene_iter_count += 1
                    else:
                        Scene_iter = iter(Scenedataloader)
                        realposter = Scene_iter.next()
                        Scene_iter_count = 2
                    bs_cur = realposter.size(0)
                    # realposter 0~1
                    netD.zero_grad()
                    errD_real = netD(realposter.to('cuda:0'))
                    errD_real.backward(one)

                    #noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
                    noise = torch.cuda.FloatTensor(bs_cur, nosie_dim, 1, 1).normal_(0, 1)
                    fakeposter = (netG(noise) + 1) / 2.0  # (0~1)
                    errD_fake = netD(fakeposter)
                    errD_fake.backward(mone)
                    optimizerD.step()



            ############################
            # (2) Update G network
            ###########################
            for p in netG.parameters():
                p.requires_grad = True
            netG.apply(act_bn)
            for p in netD.parameters():
                p.requires_grad = False
            netD.apply(fix_bn)

            bs_cur = 20
            #noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
            noise = torch.cuda.FloatTensor(bs_cur, nosie_dim, 1, 1).normal_(0, 1)
            GAN_out = (netG(noise) + 1) / 2.

            D_loss = netD(GAN_out)


            D_loss.backward()
            if iter_i % G_iter_num == 0:
                optimizerG.step()
                optimizerG.zero_grad()

            if iter_i % G_iter_num == 0:
                logger.info(f"epoch: {epoch_i}, iter: {iter_i}, errD_real: {float(errD_real)}, errD_fake: {float(errD_fake)}, errG: {float(D_loss)}")
            if iter_i % 1000 == 0:
                vis_img = torch.flip(GAN_out.cpu().detach(), [1])  # (m,3,256,256) 0~1 rgb
                torchvision.utils.save_image(vis_img, os.path.join(cfg.work_dir, f"e{epoch_i + 1}i{iter_i}.jpg"))

            torch.cuda.empty_cache()

        # 保存海报
        if (epoch_i + 1) % 10 == 0:
            torch.save(netG.state_dict(), os.path.join(cfg.work_dir, f'netG_epoch{epoch_i + 1}.pth'))
            torch.save(netD.state_dict(), os.path.join(cfg.work_dir, f'netD_epoch{epoch_i + 1}.pth'))


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
    print(len(train_dataset))

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 64, 128
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load('Spoofing3D/work_dir/GAN55-bevdet-r50-pure_advobj/netG_epoch3.pth'))
    netG.eval()

    if 0:
        noise = torch.zeros(500, nosie_dim, 1, 1).normal_(0, 1)
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear", align_corners=True)  # (m,3,200,300) 0~1 bgr
        image_poster = torch.flip(poster.clone(), [1])  # RGB
        torchvision.utils.save_image(image_poster, 'poster.jpg')
        assert False


    noise = torch.zeros(10, nosie_dim, 1, 1).normal_(0, 1)
    with torch.no_grad():
        poster = (netG(noise)+1)/2. #bgr
    poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear", align_corners=True)#(m,3,200,300) 0~1 bgr
    #torch.save(poster, 'poster.pth')
    image_poster = torch.flip(poster.clone(), [1])  # RGB
    torchvision.utils.save_image(image_poster, 'poster.jpg')
    poster = poster.to('cuda:0')


    #poster = torch.from_numpy((cv2.imread('Spoofing3D/init_poster.png')/255.0).astype(np.float32)).permute(2,0,1).unsqueeze(0).to('cuda:0')
    #poster = torch.load('poster.pth')[4].unsqueeze(0).to('cuda:0')

    ind = 777
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
        #put_poster_on_batch_inputs_eval_cross_cam(poster, data)#, spoof_cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])  # 将poster作用到img上
        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
    else:
        put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8)

    if 0: #only add points for BEVFusion
        default_lwh = (4., 1.8, 1.6)
        point_num = 20

        fake_gt = spoofed_gt[len(ori_gt):, :]
        print("spoof num = ", len(fake_gt))
        injected_point = []
        for si in range(len(fake_gt)):
            x_coord = torch.rand(point_num, 1) * default_lwh[0]*0.1 - default_lwh[0] / 2
            y_coord = torch.rand(point_num, 1) * default_lwh[1] - default_lwh[1] / 2
            z_coord = torch.rand(point_num, 1) * default_lwh[2] - default_lwh[2] / 2
            point_coord = torch.cat([x_coord, y_coord, z_coord], dim=1)
            point_coord = rotate_points_along_z(point_coord.unsqueeze(0), torch.Tensor([fake_gt[si,6]])).squeeze(0)
            point_coord = point_coord + fake_gt[si,:3].unsqueeze(0)
            point_coord[:,2] += default_lwh[2] / 2
            injected_point.append(point_coord.cuda())
        injected_point = torch.cat(injected_point, dim=0) #(m,3)
        cur_point = data['points'][0][0]
        injected_point = torch.cat([injected_point, cur_point.data[:len(injected_point),3:4], torch.zeros(len(injected_point),1).cuda()], dim=1)
        cur_point.data = torch.cat([cur_point.data, injected_point], dim=0)
        data['points'][0][0] = cur_point

    points = data['points'][0][0].data.cpu().numpy()
    points.tofile('points.bin')




    data['gt_bboxes_3d'][0][0].corners.numpy().tofile('anno.bin')
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    #print(result)

    predict_score = result[0]['pts_bbox']['scores_3d']
    mask = predict_score>0.15
    predicted_box = result[0]['pts_bbox']['boxes_3d']
    predicted_box.corners[mask].numpy().tofile('predict_anno.bin')


    if not is_4D:
        vis_imgs = []
        for i in range(6):
            img = data['img_inputs'][0][0][0][i].cpu()
            img = mmlabDeNormalize(img)  # rgb (H,W,3)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],sigle_data_dict['img_inputs'][0][5][i], 256, 704)
            #img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

            vis_imgs.append(img)

        vis_imgs = np.vstack(vis_imgs)
        cv2.imwrite('demo.png', vis_imgs)
    else:
        cam_total = []
        gt_coner = predicted_box.corners[mask]
        for cam_i in range(6):
            cam_col = []
            for adj_i in range(9):
                img = data['img_inputs'][0][0][0][cam_i * 9 + adj_i, :, :, :].cpu()
                img = mmlabDeNormalize(img)  # (3,H,W) rgb
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr

                img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                       data['img_inputs'][0][1][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][2][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][3][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][4][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][5][0, adj_i * 6 + cam_i].cpu(),
                                                       256, 704)

                cam_col.append(img)
            cam_total.append(np.vstack(cam_col))
        cam_total = np.hstack(cam_total)
        cv2.imwrite('demo.png', cam_total)

def inference_whole(is_4D=False):
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
    print(len(train_dataset))

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 64, 128
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load('Spoofing3D/work_dir/GAN44/netG_epoch16.pth'))
    netG.eval()

    from tqdm import tqdm
    for ind in tqdm(range(len(train_dataset))):
        noise = torch.zeros(2, nosie_dim, 1, 1).normal_(0, 1)
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
        poster = poster.to('cuda:0')

        sigle_data_dict = train_dataset[ind]
        sample_info = train_dataset.data_infos[ind]

        from mmcv.parallel import collate, scatter
        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        if not is_4D:
            ori_gt = data['gt_bboxes_3d'][0][0].tensor
            put_poster_on_batch_inputs_eval(poster, data)  # 将poster作用到img上
            #put_poster_on_batch_inputs_eval_cross_cam(poster, data)#, spoof_cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])  # 将poster作用到img上
            spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8)


        #data['gt_bboxes_3d'][0][0].corners.numpy().tofile('anno.bin')
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)


        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score>0.2
        predicted_box = result[0]['pts_bbox']['boxes_3d']
        #predicted_box.corners[mask].numpy().tofile('predict_anno.bin')


        if not is_4D:
            vis_imgs = []
            for i in range(6):
                img = data['img_inputs'][0][0][0][i].cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],sigle_data_dict['img_inputs'][0][5][i], 256, 704)
                #img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

                vis_imgs.append(img)

            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite(os.path.join(args.work_dir, f"{ind:0>6}" + '.jpg'), vis_imgs)
        else:
            cam_total = []
            gt_coner = predicted_box.corners[mask]
            for cam_i in range(6):
                cam_col = []
                for adj_i in range(9):
                    img = data['img_inputs'][0][0][0][cam_i * 9 + adj_i, :, :, :].cpu()
                    img = mmlabDeNormalize(img)  # (3,H,W) rgb
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr

                    img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                           data['img_inputs'][0][1][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][2][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][3][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][4][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][5][0, adj_i * 6 + cam_i].cpu(),
                                                           256, 704)

                    cam_col.append(img)
                cam_total.append(np.vstack(cam_col))
            cam_total = np.hstack(cam_total)
            cv2.imwrite(os.path.join(args.work_dir, f"{ind:0>6}" + '.jpg'), cam_total)

def inference_whole_poster(is_4D=False):
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
    print(len(train_dataset))

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 64, 128
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load('Spoofing3D/work_dir/GAN51-4d-swint/netG_epoch16.pth'))
    netG.eval()
    noise = torch.zeros(500, nosie_dim, 1, 1).normal_(0, 1)
    with torch.no_grad():
        poster = (netG(noise) + 1) / 2.  # bgr
    poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
    image_poster = torch.flip(poster.clone(), [1])  # RGB
    torchvision.utils.save_image(image_poster, 'poster.jpg')
    torch.save(poster, 'GAN51-poster.pth')
    poster = poster.to('cuda:0')

    from tqdm import tqdm
    for pi in tqdm(range(poster.size(0))):
        ind = 1292

        sigle_data_dict = train_dataset[ind]
        sample_info = train_dataset.data_infos[ind]

        from mmcv.parallel import collate, scatter
        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        if not is_4D:
            ori_gt = data['gt_bboxes_3d'][0][0].tensor
            put_poster_on_batch_inputs_eval(poster, data,use_next_poster=False)  # 将poster作用到img上
            #put_poster_on_batch_inputs_eval_cross_cam(poster, data)#, spoof_cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])  # 将poster作用到img上
            spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8,use_next_poster=False)


        #data['gt_bboxes_3d'][0][0].corners.numpy().tofile('anno.bin')
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)


        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score>0.15
        predicted_box = result[0]['pts_bbox']['boxes_3d']
        #predicted_box.corners[mask].numpy().tofile('predict_anno.bin')


        if not is_4D:
            vis_imgs = []
            for i in range(6):
                img = data['img_inputs'][0][0][0][i].cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],sigle_data_dict['img_inputs'][0][5][i], 256, 704)
                #img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

                vis_imgs.append(img)

            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite(os.path.join(args.work_dir, 'demo' + str(pi) + '.jpg'), vis_imgs)
        else:
            cam_total = []
            gt_coner = predicted_box.corners[mask]
            for cam_i in range(6):
                cam_col = []
                for adj_i in range(9):
                    img = data['img_inputs'][0][0][0][cam_i * 9 + adj_i, :, :, :].cpu()
                    img = mmlabDeNormalize(img)  # (3,H,W) rgb
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr

                    img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                           data['img_inputs'][0][1][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][2][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][3][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][4][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][5][0, adj_i * 6 + cam_i].cpu(),
                                                           256, 704)

                    cam_col.append(img)
                cam_total.append(np.vstack(cam_col))
            cam_total = np.hstack(cam_total)
            cv2.imwrite(os.path.join(args.work_dir, 'demo' + str(pi) + '.jpg'), cam_total)
        poster = poster[1:, :, :, :]

def eval(G_dir,score_thr=0.1, iou_thr=[0.1,0.3,0.5,0.7], center_thr=[0.5,1,2,3], is_4D=False, is_add_point=False):
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

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(G_dir))
    netG.to('cuda:0')
    netG.eval()



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

        noise = torch.zeros(2, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
        #poster = torch.rand(poster.size()).to('cuda:0')
        #poster = torch.from_numpy((cv2.imread('Spoofing3D/init_poster.png') / 255.0).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to('cuda:0')

        #poster_collection = torch.load('BF-SwinT-rgb-selected.pth')
        #poster_collection = torch.flip(poster_collection, [1])
        #poster = poster_collection[np.random.randint(poster_collection.size(0))].unsqueeze(0).to('cuda:0')


        #poster = torch.load('Spoofing3D/work_dir/try24/poster_8.pth').to('cuda:0')
        #poster = poster.permute(2,0,1).unsqueeze(0)


        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        ori_gt = data['gt_bboxes_3d'][0][0].tensor

        if not is_4D:
            put_poster_on_batch_inputs_eval(poster, data, mask_aug=False,use_next_poster=False)  # 将poster作用到img上
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8,use_next_poster=False)  # 将poster作用到img上

        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        if len(ori_gt) == len(spoofed_gt): continue
        fake_gt = spoofed_gt[len(ori_gt):,:]
        valid_spoof += len(fake_gt)

        if is_add_point and len(fake_gt)>0 :  # only add points for BEVFusion
            default_lwh = (4., 1.8, 1.6)
            point_num = 20

            injected_point = []
            for si in range(len(fake_gt)):
                x_coord = torch.rand(point_num, 1) * default_lwh[0] * 0.1 - default_lwh[0]*0.1 / 2
                y_coord = torch.rand(point_num, 1) * default_lwh[1] - default_lwh[1] / 2
                z_coord = torch.rand(point_num, 1) * default_lwh[2] - default_lwh[2] / 2
                point_coord = torch.cat([x_coord, y_coord, z_coord], dim=1)
                point_coord = rotate_points_along_z(point_coord.unsqueeze(0), torch.Tensor([fake_gt[si, 6]])).squeeze(0)
                point_coord = point_coord + fake_gt[si, :3].unsqueeze(0)
                point_coord[:, 2] += default_lwh[2] / 2
                injected_point.append(point_coord.cuda())
            injected_point = torch.cat(injected_point, dim=0)  # (m,3)
            cur_point = data['points'][0][0]
            injected_point = torch.cat(
                [injected_point, cur_point.data[:len(injected_point), 3:4], torch.zeros(len(injected_point), 1).cuda()],
                dim=1)
            cur_point.data = torch.cat([cur_point.data, injected_point], dim=0)
            data['points'][0][0] = cur_point



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

        if jj % 100 == 0:
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

def eval_two_stage(G_dir,score_thr=0.1, iou_thr=[0.1,0.3,0.5,0.7], center_thr=[0.5,1,2,3], is_4D=False, is_add_point=False):
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
    for p in model.parameters():
        p.requires_grad = False


    datasets = [build_dataset(cfg.data.val)]
    eval_dataset = datasets[0]
    print(f"测试集共包含{len(eval_dataset)}帧")


    train_dataset = build_dataset(cfg.data.test)
    assert len(eval_dataset) == len(train_dataset)

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(G_dir))
    netG.to('cuda:0')
    netG.eval()
    for p in netG.parameters():
        p.requires_grad = False





    total_test_frame = 1000#1000
    valid_spoof = 0
    success_spoof_iou = {}
    success_spoof_centerDistant = {}
    for thr in iou_thr:
        success_spoof_iou['iou_%s'%str(thr)] = 0
    for thr in center_thr:
        success_spoof_centerDistant["center_%s"%str(thr)] = 0


    inds = np.random.choice(list(range(len(eval_dataset))), total_test_frame, replace=False)

    #noise = torch.zeros(1, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
    #noise = nn.Parameter(noise, requires_grad=True)
    #optimizer_noise = torch.optim.Adam([noise], lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    for jj,ind in enumerate(inds):
        #ind = 2155
        sigle_data_dict = train_dataset[ind]

        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        '''更新noise'''
        noise = torch.zeros(1, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
        noise = nn.Parameter(noise, requires_grad=True)
        optimizer_noise = torch.optim.Adam([noise], lr=0.1,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

        for noise_update_iter in range(30):
            poster = (netG(noise) + 1) / 2.  # bgr
            poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear", align_corners=True)  # (1,3,200,300) 0~1 bgr

            input_data = copy.deepcopy(data)
            if not is_4D:
                if noise_update_iter == 0:
                    location_dict = maskGT_put_poster_on_batch_inputs_v2(poster, input_data, mask_aug=False, location_dict=None,use_next_poster=False)
                    location_dict=None
                else:
                    empty_ = maskGT_put_poster_on_batch_inputs_v2(poster, input_data, mask_aug=False, location_dict=copy.deepcopy(location_dict),use_next_poster=False)
            else:
                maskGT_put_poster_on_batch_inputs_4D(poster, input_data, is_bilinear=True, num_adj=8, use_next_poster=False)

            '''
            vis_imgs = []
            frame_i = 0
            for i in range(6):
                img = input_data['img_inputs'][0][frame_i][i].detach().cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = draw_box_from_batch(img, input_data, frame_i, i)
                vis_imgs.append(img)

            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite('demo1.png', vis_imgs)
            assert False
            '''


            if len(input_data['gt_bboxes_3d'][0].tensor) == 0:
                break
            output = model(return_loss=True, **input_data)  # 前向传播计算loss
            adv_loss = 0
            for k in output.keys():
                adv_loss += output[k]
            #print(noise_update_iter, adv_loss)
            optimizer_noise.zero_grad()
            adv_loss.backward()
            optimizer_noise.step()


        '''eval'''
        sigle_data_dict = eval_dataset[ind]
        sample_info = eval_dataset.data_infos[ind]

        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr


        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        ori_gt = data['gt_bboxes_3d'][0][0].tensor

        if not is_4D:
            put_poster_on_batch_inputs_eval(poster, data, mask_aug=False,use_next_poster=False)  # 将poster作用到img上
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8,use_next_poster=False,only_apply_cur_frame=False)  # 将poster作用到img上

        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        if len(ori_gt) == len(spoofed_gt): continue
        fake_gt = spoofed_gt[len(ori_gt):,:]
        valid_spoof += len(fake_gt)

        if is_add_point and len(fake_gt)>0 :  # only add points for BEVFusion
            default_lwh = (4., 1.8, 1.6)
            point_num = 20

            injected_point = []
            for si in range(len(fake_gt)):
                x_coord = torch.rand(point_num, 1) * default_lwh[0] * 0.1 - default_lwh[0]*0.1 / 2
                y_coord = torch.rand(point_num, 1) * default_lwh[1] - default_lwh[1] / 2
                z_coord = torch.rand(point_num, 1) * default_lwh[2] - default_lwh[2] / 2
                point_coord = torch.cat([x_coord, y_coord, z_coord], dim=1)
                point_coord = rotate_points_along_z(point_coord.unsqueeze(0), torch.Tensor([fake_gt[si, 6]])).squeeze(0)
                point_coord = point_coord + fake_gt[si, :3].unsqueeze(0)
                point_coord[:, 2] += default_lwh[2] / 2
                injected_point.append(point_coord.cuda())
            injected_point = torch.cat(injected_point, dim=0)  # (m,3)
            cur_point = data['points'][0][0]
            injected_point = torch.cat(
                [injected_point, cur_point.data[:len(injected_point), 3:4], torch.zeros(len(injected_point), 1).cuda()],
                dim=1)
            cur_point.data = torch.cat([cur_point.data, injected_point], dim=0)
            data['points'][0][0] = cur_point


        with torch.no_grad(): #前向推理
            result = model(return_loss=False, rescale=True, **data)

        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score > score_thr
        predicted_box = result[0]['pts_bbox']['boxes_3d'].tensor[mask] #(N,9)
        if len(predicted_box) ==0 : continue

        '''
        vis_imgs = []
        for i in range(6):
            img = data['img_inputs'][0][0][0][i].cpu()
            img = mmlabDeNormalize(img)  # rgb (H,W,3)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            img = visulize_3dbox_to_cam(img, result[0]['pts_bbox']['boxes_3d'].corners[mask], sample_info, cams[i],
                                        sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i],
                                        256, 704)
            # img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

            vis_imgs.append(img)

        vis_imgs = np.vstack(vis_imgs)
        cv2.imwrite('demo2.png', vis_imgs)
        assert False
        '''





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

        if jj % 100 == 0:
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

def eval_LPIPS(G_dir, is_4D=False):
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

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(G_dir))
    netG.to('cuda:0')
    netG.eval()

    lpips_model = lpips.LPIPS(net="alex")

    total_test_frame = 1000
    lpips_total = 0
    valid_cal = 0



    inds = np.random.choice(list(range(len(train_dataset))), total_test_frame, replace=False)
    #inds = [2937]
    for jj,ind in enumerate(inds):
        sigle_data_dict = train_dataset[ind]
        #sample_info = train_dataset.data_infos[ind]

        if not is_4D:
            #scene_image = mmlabDeNormalize(sigle_data_dict['img_inputs'][0][0][1,:,:,:]) #rgb (H,W,3) 0~255 numpy
            #cv2.cvtColor(scene_image, cv2.COLOR_RGB2BGR, scene_image)
            #cv2.imwrite('demo0.png', scene_image)

            ori_image = mmlabDeNormalize(sigle_data_dict['img_inputs'][0][0][1,:,:,224:480]) #rgb (H,W,3) 0~255 numpy
            #vis_ori = copy.deepcopy(ori_image)
            #cv2.cvtColor(vis_ori, cv2.COLOR_RGB2BGR, vis_ori)
            #cv2.imwrite('demo1.png', vis_ori)
            ori_image = (torch.from_numpy(ori_image).permute(2,0,1)/255.).unsqueeze(0).float() #rgb (1,3,h,w) 0~1
        else:
            ori_image = mmlabDeNormalize(sigle_data_dict['img_inputs'][0][0][1*9+0, :, :, 224:480])  # rgb (H,W,3) 0~255 numpy
            #cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR, ori_image)
            #cv2.imwrite('demo1.png', ori_image)
            ori_image = (torch.from_numpy(ori_image).permute(2, 0, 1) / 255.).unsqueeze(0).float()  # rgb (1,3,h,w) 0~1

        noise = torch.zeros(2, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
        #poster = torch.rand(poster.size()).to('cuda:0')
        #poster = torch.from_numpy((cv2.imread('Spoofing3D/init_poster.png') / 255.0).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to('cuda:0')

        #poster_collection = torch.load('BF-SwinT-rgb-selected.pth')
        #poster_collection = torch.flip(poster_collection, [1])
        #poster = poster_collection[np.random.randint(poster_collection.size(0))].unsqueeze(0).to('cuda:0')

        #poster = torch.load('Spoofing3D/work_dir/try24/poster_8.pth').to('cuda:0')
        #poster = poster.permute(2, 0, 1).unsqueeze(0)


        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        ori_gt = data['gt_bboxes_3d'][0][0].tensor

        if not is_4D:
            put_poster_on_batch_inputs_eval(poster, data, mask_aug=False,use_next_poster=False,spoof_cams=['CAM_FRONT'])  # 将poster作用到img上
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8,use_next_poster=False,spoof_cams=['CAM_FRONT'])  # 将poster作用到img上

        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        if len(ori_gt) == len(spoofed_gt): continue

        valid_cal+=1
        if not is_4D:
            spoof_image = data['img_inputs'][0][0][0][1].cpu()
            spoof_image = mmlabDeNormalize(spoof_image[:, :, 224:480])  # rgb (H,W,3) 0~255 numpy
            #vis_spoof = copy.deepcopy(spoof_image)
            #cv2.cvtColor(vis_spoof, cv2.COLOR_RGB2BGR, vis_spoof )
            #cv2.imwrite('demo2.png', vis_spoof)
            #assert False
            spoof_image = (torch.from_numpy(spoof_image).permute(2, 0, 1) / 255.).unsqueeze(0).float()  # rgb (1,3,h,w) 0~1
        else:
            spoof_image = data['img_inputs'][0][0][0][1*9+0].cpu()
            spoof_image = mmlabDeNormalize(spoof_image[:, :, 224:480])  # rgb (H,W,3) 0~255 numpy
            #cv2.cvtColor(spoof_image, cv2.COLOR_RGB2BGR, spoof_image )
            #cv2.imwrite('demo2.png', spoof_image)
            spoof_image = (torch.from_numpy(spoof_image).permute(2, 0, 1) / 255.).unsqueeze(0).float()  # rgb (1,3,h,w) 0~1

        lp_distance = lpips_model(ori_image, spoof_image)
        lpips_total += lp_distance




        print("\r",  f"{jj}/{total_test_frame}", f"valid spoof = {valid_cal}", f"Avg. LPIPS= {lpips_total/valid_cal}" ,end="")
    print(f"Avg. LPIPS= {lpips_total / valid_cal}")

















if __name__ == '__main__':
    is_BEVDet4D = True

    #Training(is_4D=is_BEVDet4D)
    #inference(is_4D=is_BEVDet4D)
    #inference_whole(is_4D=is_BEVDet4D)
    #eval(G_dir='Spoofing3D/work_dir/GAN58-bevdet4d-swint-pure_advobj/netG_epoch1.pth',score_thr=0.1, iou_thr=[0.1,0.3,0.5,0.7], center_thr=[0.5,1,1.5,2,3],is_4D=is_BEVDet4D,is_add_point=False)
    #Training_pure_GAN()
    eval_two_stage(G_dir='Spoofing3D/work_dir/GAN50-4d-res50/netG_epoch16.pth', score_thr=0.1, iou_thr=[0.1, 0.3, 0.5, 0.7], center_thr=[0.5, 1, 1.5, 2, 3], is_4D=is_BEVDet4D, is_add_point=False)

    #inference_whole_poster(is_4D=is_BEVDet4D)
    #eval_LPIPS(G_dir='Spoofing3D/work_dir/GAN58-bevdet4d-swint-pure_advobj/netG_epoch1.pth', is_4D=is_BEVDet4D)

