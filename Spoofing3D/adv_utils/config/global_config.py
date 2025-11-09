import numpy as np

mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32) / 255  # bgr下
std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32) / 255
img_h, img_w = 256, 704
cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4,
               'CAM_BACK_RIGHT': 5}
camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90,
                   'CAM_BACK_RIGHT': -1}
random_an = 15
spoof_cams=['CAM_FRONT', 'CAM_BACK']
sample_range = (6, 12)
default_lwh = (4., 1.8, 1.6)
physical_lw = (3.0, 2.0)
max_search_num = 20
l, w = physical_lw[0], physical_lw[1]