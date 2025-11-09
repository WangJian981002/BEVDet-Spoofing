#将图像中的gt bbox mask掉
from matplotlib.path import Path
import random

import numpy as np
import torch
from mmcv.ops import box_iou_rotated

from Spoofing3D.adv_utils.common_utils import rotate_points_along_z
from Spoofing3D.adv_utils.config.global_config import *



def poster_corner_inlidar(fake_box,z_bottle):
    fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
    #print(fake_3d_box)

    '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
    poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                  [l / 2, -w / 2, z_bottle],
                                  [-l / 2, -w / 2, z_bottle],
                                  [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

    poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
    poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
    return fake_3d_box,poster_corner
def poster_3d_position(spoofcam):
    r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
    an = (2 * np.random.rand() - 1) * (random_an * np.pi / 180.) + camcenter_angle[
        spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
    cx = r * np.cos(an)
    cy = r * np.sin(an)
    yaw = (2 * np.random.rand() - 1) * (random_an * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
    fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
    fake_box = torch.from_numpy(fake_box)

    return fake_box

def maskGT_put_poster_on_batch_inputs(leaning_poster, batch_inputs, spoof_cams=spoof_cams, is_bilinear=False, mask_aug=False):

    poster_w, poster_l = leaning_poster.size()[:2]
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
                fake_box = poster_3d_position(spoofcam)
                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            # car_z = gt_box[batch_inputs['gt_labels_3d'][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1. : continue #防止飘在空中的情况

            fake_3d_box,poster_corner = poster_corner_inlidar(fake_box,z_bottle)

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
                    bound_pixel = 50
                    start_w = np.random.choice(poster_w-bound_pixel)
                    start_l = np.random.choice(poster_l-bound_pixel)
                    end_w = min(start_w+200, poster_w)
                    end_l = min(start_l+200, poster_l)
                    leaning_poster_clone[start_w:end_w, start_l:end_l,:] = 0

            else:
                leaning_poster_clone = leaning_poster


            if is_bilinear:
                index_l = torch.clip((lidar_inner_points[:, 0] / l)*2-1, min=-1, max=1)
                index_w = torch.clip(((w - lidar_inner_points[:, 1]) / w)*2-1, min=-1, max=1)
                grid = torch.cat([index_l.unsqueeze(-1), index_w.unsqueeze(-1)], dim=1).unsqueeze(0).unsqueeze(0) #(1,1,Nin,2)
                selected_color = torch.nn.functional.grid_sample(leaning_poster_clone.permute(2,0,1).unsqueeze(0), grid.cuda(), mode='bilinear', align_corners=True) #(1,3,1,Nin)
                selected_color = selected_color.squeeze().permute(1,0)

            else :
                index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                selected_color = leaning_poster_clone[index_w, index_l, :] #(Nin, 3) bgr 0~1 gpu

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

def maskGT_put_poster_on_batch_inputs_4D(leaning_poster, batch_inputs, spoof_cams=spoof_cams, is_bilinear=False, num_adj=8):

    poster_w, poster_l = leaning_poster.size()[:2]
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
                fake_box = poster_3d_position(spoofcam)

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

            fake_3d_box,poster_corner = poster_corner_inlidar(fake_box,z_bottle)

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
                    selected_color = torch.nn.functional.grid_sample(leaning_poster.permute(2,0,1).unsqueeze(0), grid.cuda(), mode='bilinear', align_corners=True) #(1,3,1,Nin)
                    selected_color = selected_color.squeeze().permute(1,0)

                else :
                    index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                    index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                    selected_color = leaning_poster[index_w, index_l, :] #(Nin, 3) bgr 0~1 gpu

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
        #只保留spoofer的gt信息
        batch_inputs['gt_bboxes_3d'][frame_i].tensor = batch_inputs['gt_bboxes_3d'][frame_i].tensor[num_obj_in_frame:,:]
        batch_inputs['gt_labels_3d'][frame_i] = batch_inputs['gt_labels_3d'][frame_i][num_obj_in_frame:].long()
