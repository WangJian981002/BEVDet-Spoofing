import cv2
import numpy as np
import torch


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