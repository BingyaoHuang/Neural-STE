'''
Neural-STE CNN models
'''

import torch
import torch.nn as nn
import copy
import kornia


# interpolation layer
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


# For geometric correction
class WarpingNet(nn.Module):
    def __init__(self, chan_in=3, out_size=(256, 256)):
        super(WarpingNet, self).__init__()
        self.name = self.__class__.__name__
        self.chan_in = chan_in  # input image channels
        self.out_size = out_size

        # Spatial transformer network (STN) with homography
        self.interp_homography = Interpolate((28, 28), 'bilinear')
        self.localization_homography = nn.Sequential(
            self.interp_homography,
            nn.Conv2d(self.chan_in, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 3 homography
        self.fc_loc_homography = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 3)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc_homography[2].weight.data.zero_()
        self.fc_loc_homography[2].bias.data.copy_(torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 1]))

    def homography_warp(self, x):
        xs = self.localization_homography(x)
        H = self.fc_loc_homography(xs.view(-1, 10 * 3 * 3)).view(-1, 3, 3)

        # warp using homography
        x = kornia.homography_warp(x, H, self.out_size)

        return x

    # Spatial transformer network (STN) forward function
    def forward(self, I):
        I_warp = self.homography_warp(I)
        return I_warp


# DehazingNet and RefineNet
class DehazingRefineNet(nn.Module):
    def __init__(self, chan_in=3, chan_out=3, degradation=''):
        super().__init__()
        self.name = self.__class__.__name__ + '_' + degradation
        self.relu = nn.ReLU()
        self.degradation = degradation
        self.chan_in = chan_in
        self.chan_out = chan_out

        # backbone (G)
        # downsample conv (encoder)
        self.d1 = self.down_conv(self.chan_in, 32, batch_norm=False)  # 128
        self.d2 = self.down_conv(32, 64)  # 64
        self.d3 = self.down_conv(64, 128)  # 32
        self.d4 = self.down_conv(128, 256)  # 16
        self.d5 = self.down_conv(256, 256)  # 8
        self.d6 = self.down_conv(256, 256)  # 4
        self.d7 = self.down_conv(256, 256)  # 2

        # bottleneck, no batch norm and relu
        self.bn = self.down_conv(256, 256, batch_norm=False, leaky_relu=False)  # 1

        # upample conv (decoder)
        self.u1 = self.up_conv(256, 256)  # 2 #
        self.u2 = self.up_conv(512, 256)  # 4 #
        self.u3 = self.up_conv(512, 256)  # 8
        self.u4 = self.up_conv(512, 256)  # 16
        self.u5 = self.up_conv(512, 128)  # 32
        self.u6 = self.up_conv(256, 64)  # 64
        self.u7 = self.up_conv(128, 32)  # 128

        # output layer for proposed "black box"
        if self.degradation in ['black_box']:
            self.out = nn.Sequential(nn.ConvTranspose2d(64, self.chan_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.Sigmoid())  # use Tanh if image is normalized to [-1, 1], otherwise relu or sigmoid

        # FA and Ft
        if self.degradation not in ['black_box']:
            # transmittance inference network
            self.Ft = nn.Sequential(nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(3, self.chan_out, kernel_size=3, stride=1, padding=1),
                                    nn.Sigmoid())  # use Tanh if image is normalized to [-1, 1], otherwise relu or sigmoid

            # reflected light inference network
            self.FA = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(16, self.chan_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid())

        # RefineNet (phi)
        if self.degradation not in ['black_box', 'no_refine']:
            self.refine_net = nn.Sequential(nn.Conv2d(self.chan_out, 32, kernel_size=7, stride=1, padding=3),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(32, self.chan_out, kernel_size=3, stride=1, padding=1),
                                            # nn.Sigmoid()) # use Tanh if image is normalized to [-1, 1], otherwise relu or sigmoid
                                            nn.Tanh())  # use Tanh if image is normalized to [-1, 1], otherwise relu or sigmoid

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_initialize_weights)

    def down_conv(self, chan_in, chan_out, kernel_size=3, batch_norm=True, leaky_relu=True):
        # downsample conv
        model = [nn.Conv2d(chan_in, chan_out, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2)]

        # batch normalization
        if batch_norm:
            model = model + [nn.BatchNorm2d(chan_out)]

        # activation
        model = model + [nn.LeakyReLU(0.2)] if leaky_relu else model + [nn.ReLU()]

        return nn.Sequential(*model)

    def up_conv(self, chan_in, chan_out, kernel_size=3, batch_norm=True):
        # upsample conv
        model = [nn.ConvTranspose2d(chan_in, chan_out, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)]

        # batch normalization
        if batch_norm:
            model = model + [nn.BatchNorm2d(chan_out)]

        return nn.Sequential(*model)

    def forward(self, I):

        # backbone (G)
        # downsample conv
        d1 = self.d1(I)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        # bottleneck
        bn = self.bn(d7)

        # upsample conov
        u1 = self.relu(torch.cat((self.u1(bn), d7), 1))
        u2 = self.relu(torch.cat((self.u2(u1), d6), 1))
        u3 = self.relu(torch.cat((self.u3(u2), d5), 1))
        u4 = self.relu(torch.cat((self.u4(u3), d4), 1))
        u5 = self.relu(torch.cat((self.u5(u4), d3), 1))
        u6 = self.relu(torch.cat((self.u6(u5), d2), 1))
        u7 = self.relu(torch.cat((self.u7(u6), d1), 1))

        # output based on degradation
        if self.degradation in ['black_box']:
            A = I * 0
            t = I * 0
            J_coarse = I * 0
            J_hat = torch.clamp(self.out(u7), max=1)
        else:
            A = torch.clamp(self.FA(u7), max=1)
            t = torch.clamp(self.Ft(u7), min=0.01, max=1)
            J_coarse = torch.clamp(torch.clamp(I - A, 0, 1) / t, min=0, max=1)

            if self.degradation == 'no_refine':
                J_hat = torch.clamp(J_coarse, min=0, max=1)
            elif self.degradation in ['', 'no_warp', 'no_A_const', 'no_J_const']:
                J_hat = torch.clamp(J_coarse + self.refine_net(J_coarse), min=0, max=1)
            else:
                raise ValueError(self.degradation)

        return J_hat, J_coarse, A, t


# WarpingNet + DehazingRefineNet
class NeuralSTE(nn.Module):
    def __init__(self, warping_net=None, dehazing_refine_net=None, degradation=''):
        super(NeuralSTE, self).__init__()
        self.degradation = degradation
        self.name = 'Neural-STE_' + degradation if degradation != '' else 'Neural-STE'

        # initialize from existing models or create new models
        if self.degradation not in ['black_box', 'no_warp']:
            self.warping_net = copy.deepcopy(warping_net.module)
        else:
            self.resize = Interpolate((256, 256), 'bilinear')
        self.dehazing_refine_net = copy.deepcopy(dehazing_refine_net.module)

    def forward(self, I):

        # geometric correction using WarpingNet
        if self.degradation not in ['black_box', 'no_warp']:
            I_warp = self.warping_net(I)
        else:
            I_warp = self.resize(I)

        # dehaze and refine
        J_hat, J_coarse, A, t = self.dehazing_refine_net(I_warp)

        return J_hat, I_warp, J_coarse, A, t