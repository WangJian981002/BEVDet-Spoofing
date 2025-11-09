import math
import numpy as np
import torch
from torch import nn

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

def get_reference_points(spatial_shapes,  # 多尺度feature map对应的h,w，shape为[num_level,2]
                         valid_ratios,  # 多尺度feature map对应的mask中有效的宽高比，shape为[B, num_levels, 2]
                         device='cpu'):
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        # 对于每一层feature map初始化每个参考点中心横纵坐标，加减0.5是确保每个初始点是在每个pixel的中心，例如[0.5,1.5,2.5, ...]
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=torch.float32, device=device))

        # 将横纵坐标进行归一化，处理成0-1之间的数
        ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)

        # 得到每一层feature map对应的reference point，即ref，shape为[B, feat_W*feat_H, 2]
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)

    # 将所有尺度的feature map对应的reference point在第一维合并，得到[2, N, 2]
    reference_points = torch.cat(reference_points_list, 1)
    # 从[2, N, 2]扩充尺度到[2, N, num_level, 2] (x,y)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points



class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.contiguous()



if __name__ == "__main__":
    '''
    #测试reference points
    spatial_shapes = [[3,5]]
    valid_ratios = torch.ones(2,1,2)
    reference_points = get_reference_points(spatial_shapes, valid_ratios, 'cpu')

    print(reference_points.size())
    print(reference_points)
    '''
    '''
    #测试位置编码
    pm = PositionEmbeddingSine(10)

    mask = torch.ones(1,4,4).bool()
    pos_embed = pm(mask)
    print(pos_embed.size())
    '''

    MSDA = MultiScaleDeformableAttention(embed_dims=256, num_levels=1)
    MSDA = MSDA.cuda()

    x = torch.randn(2,256,180,180).cuda()
    y = torch.randn(2,256,180,180).cuda()

    B,C,_,_ = x.size()
    query = x.permute(0,2,3,1).contiguous().view(B,-1,C).permute(1,0,2) #(hw,B,C)
    value = y.permute(0,2,3,1).contiguous().view(B,-1,C).permute(1,0,2) #(hw,B,C)

    PE = PositionEmbeddingSine(128)
    pos_embed = PE(torch.ones(2,180,180).bool()) #(B, C, H, W)
    pos_embed = pos_embed.permute(0,2,3,1).contiguous().view(B,-1,C).permute(1,0,2).cuda()#(hw,B,C)

    reference_points = get_reference_points([[180,180]], torch.ones(2,1,2)).cuda()#(B, hw, 1, 2)

    spatial_shapes = torch.Tensor([[180, 180]]).long()

    level_start_index = torch.Tensor([0]).long()

    out = MSDA(query=query, value=value, query_pos=pos_embed, reference_points=reference_points, spatial_shapes=spatial_shapes.cuda(), level_start_index=level_start_index.cuda())
    print(out.size())