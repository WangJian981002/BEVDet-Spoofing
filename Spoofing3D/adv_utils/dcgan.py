import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.parallel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class SceneSet(Dataset):
    def __init__(self,data_root='',to_rgb=False,imgsize=[256,256]):
        super().__init__()
        self.to_rgb = to_rgb
        self.imgsize = imgsize

        img_list = os.listdir(data_root)
        self.img_dir = []
        for name in img_list:
            self.img_dir.append(os.path.join(data_root, name))

    def __len__(self):
        return len(self.img_dir)


    def __getitem__(self, idx):
        img = cv2.imread(self.img_dir[idx])
        img = cv2.resize(img, self.imgsize)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose((img/255.0), (2,0,1))

        return np.ascontiguousarray(img).astype(np.float32)



def weights_init(m):
    """initialize GAN network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DCGAN_D_CustomAspectRatio(nn.Module):
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0):
        super(DCGAN_D_CustomAspectRatio, self).__init__()
        assert isize[0] % 16 == 0 and isize[1] % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize[0] x isize[1]
        main.add_module(f'initial:{nc}-{ndf}:conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module(f'initial:{ndf}:relu', nn.LeakyReLU(0.2, inplace=True))
        csize = [isize[0] // 4, isize[1] // 4]
        cndf = ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-layers-{t}:{cndf}:conv', nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(f'extra-layers-{t}:{cndf}:batchnorm', nn.BatchNorm2d(cndf))
            main.add_module(f'extra-layers-{t}:{cndf}:relu', nn.LeakyReLU(0.2, inplace=True))

        while csize[0] > 4 and csize[1] > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(f'pyramid:{in_feat}-{out_feat}:conv', nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(f'pyramid:{out_feat}:batchnorm', nn.BatchNorm2d(out_feat))
            main.add_module(f'pyramid:{out_feat}:relu', nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = [csize[0] // 2, csize[1] // 2]

        # state size. K x 4 x 4
        main.add_module(f'final:{cndf}-{1}:conv', nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):

        output = self.main(input)

        output = output.mean(0)
        return output.view(1)

class DCGAN_G_CustomAspectRatio(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DCGAN_G_CustomAspectRatio, self).__init__()

        assert isize[0] % 16 == 0 and isize[1] % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize_w= ngf // 2, 8
        csize_w = isize[0]
        while tisize_w != csize_w:
            cngf = cngf * 2
            tisize_w = tisize_w * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(f'initial:{nz}-{cngf}:convt',nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module(f'initial:{cngf}:batchnorm',nn.BatchNorm2d(cngf))
        main.add_module(f'initial:{cngf}:relu',nn.ReLU(True))

        csize_w, csize_h, cndf = 8, 3, cngf
        while csize_w < isize[0] // 2:   # Make sure we stop when width is close to isize
            main.add_module(f'pyramid:{cngf}-{cngf // 2}:convt', nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module(f'pyramid:{cngf // 2}:batchnorm', nn.BatchNorm2d(cngf // 2))
            main.add_module(f'pyramid:{cngf // 2}:relu', nn.ReLU(True))
            cngf = cngf // 2
            csize_w = csize_w * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra-layers-{t}:{cngf}:conv', nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module(f'extra-layers-{t}:{cngf}:batchnorm', nn.BatchNorm2d(cngf))
            main.add_module(f'extra-layers-{t}:{cngf}:relu', nn.ReLU(True))

        main.add_module(f'final:{cngf}-{nc}:convt', nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        self.main = main

        self.active = nn.Tanh()

    def forward(self, input):
        output = self.main(input)
        output = self.active(output)

        return output




if __name__ == '__main__':
    
    rowPatch_size = [256,256 ] #补丁输出尺寸为此一半
    nc = 3
    ndf = 64
    ngf = 64
    n_extra_layers = 0
    nz = 100


    test_tensor = torch.rand(8, 3, 128, 128)
    test_dis_origin = DCGAN_D_CustomAspectRatio(rowPatch_size, nz, nc, ndf, n_extra_layers)
    print(test_dis_origin)
    output_dcgan = test_dis_origin(test_tensor)
    print(output_dcgan.size())

    noise = torch.FloatTensor(1, nz, 1, 1)
    test_gen_origin = DCGAN_G_CustomAspectRatio(rowPatch_size, nz, nc, ngf, n_extra_layers)
    print(test_gen_origin)
    output_gen_dcgan = test_gen_origin(noise)
    print(output_gen_dcgan.size())
    '''
    trainset = SceneSet('data/Background_scene/forest', imgsize=[256,256])
    print(len(trainset))
    Scenedataloader = DataLoader(trainset, batch_size=64, shuffle=True)
    print(len(Scenedataloader))


    dataiter = iter(Scenedataloader)
    data = dataiter.next()
    print(data.size())
    data = dataiter.next()
    print(data.size())
    data = dataiter.next()
    print(data.size())
    data = dataiter.next()
    print(data.size())
    '''
