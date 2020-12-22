'''
Useful helper functions
'''

import os
from os.path import join as fullfile
import platform
import numpy as np
import cv2 as cv
import math
import random
import torch
import torch.nn as nn
import torchvision.transforms
import pytorch_ssim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# for visualization
import visdom

vis = visdom.Visdom(port=8098, use_incoming_socket=False)  # default is 8097
assert vis.check_connection(), 'Visdom: No connection, start visdom first!'

# use qt5agg backend for remote interactive interpreter plot below
import matplotlib as mpl

# backend
mpl.use('Qt5Agg')
# mpl.use('TkAgg')

# disable toolbar and set background to black for full screen
mpl.rcParams['toolbar'] = 'None'
mpl.rcParams['figure.facecolor'] = 'black'

import matplotlib.pyplot as plt  # restart X11 session if it hangs (MobaXterm in my case)


# Use Pytorch multi-threaded dataloader and opencv to load image faster
class SimpleDataset(Dataset):
    """Simple dataset."""

    def __init__(self, data_root, index=None, size=None):
        self.data_root = data_root
        self.size = size

        # img list
        img_list = sorted(os.listdir(data_root))
        if index is not None: img_list = [img_list[x] for x in index]

        self.img_names = [fullfile(self.data_root, name) for name in img_list]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        assert os.path.isfile(img_name), img_name + ' does not exist'
        im = cv.imread(self.img_names[idx])

        # resize image if size is specified
        if self.size is not None:
            im = cv.resize(im, self.size[::-1])
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        return im


# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# read images using multi-thread
def readImgsMT(img_dir, size=None, index=None, gray_scale=False, normalize=False):
    img_dataset = SimpleDataset(img_dir, index=index, size=size)
    data_loader = DataLoader(img_dataset, batch_size=len(img_dataset), shuffle=False, drop_last=False, num_workers=4)

    for i, imgs in enumerate(data_loader):
        # imgs.permute((0, 3, 1, 2)).to('cpu', dtype=torch.float32)/255
        # convert to torch.Tensor
        imgs = imgs.permute((0, 3, 1, 2)).float().div(255)

        if gray_scale:
            imgs = 0.2989 * imgs[:, 0] + 0.5870 * imgs[:, 1] + 0.1140 * imgs[:, 2]  # same as MATLAB rgb2gray and OpenCV cvtColor
            imgs = imgs[:, None]

        # normalize to [-1, 1], should improve model convergence in early training stages.
        if normalize:
            imgs = (imgs - 0.5) / 0.5

        return imgs


# figure and show different type of tensors or ndarrays
def fs(inputData, title=None):
    inputData = inputData.squeeze()
    if type(inputData) is np.ndarray:
        im = inputData
    elif type(inputData) is torch.Tensor:
        F_tensor_to_image = torchvision.transforms.ToPILImage()

        if inputData.requires_grad:
            inputData = inputData.detach()

        if inputData.device.type == 'cuda':
            if inputData.ndimension() == 2:
                im = inputData.squeeze().cpu().numpy()
            else:
                im = F_tensor_to_image(inputData.squeeze().cpu())
        else:
            if inputData.ndimension() == 2:
                im = inputData.numpy()
            else:
                im = F_tensor_to_image(inputData.squeeze())

    # remove white paddings
    fig = plt.figure()
    # fig.canvas.window().statusBar().setVisible(False)

    # display image
    ax = plt.imshow(im, interpolation='bilinear')
    ax = plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if title is not None:
        plt.title(title)
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, hspace=0, wspace=0)
    plt.show()
    return fig


# from torchvision.utils.make_grid, but row and col are switched
def make_grid_transposed(tensor, nrow=8, padding=2,
                         normalize=False, irange=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        irange (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if irange is not None:
            assert isinstance(irange, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, irange)
        else:
            norm_range(tensor, irange)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    ymaps = min(nrow, nmaps)
    xmaps = int(math.ceil(float(nmaps) / ymaps))

    width, height = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0

    for x in range(xmaps):
        for y in range(ymaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


# save 4D np.ndarray or torch tensor to image files
def saveImgs(inputData, dir, idx=0):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if type(inputData) is torch.Tensor:
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    if imgs.dtype == 'float32':
        imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    else:
        imgs = imgs[:, :, :, ::-1]  # convert to BGR and uint8 for opencv
    for i in range(imgs.shape[0]):
        file_name = 'img_{:04d}.png'.format(i + 1 + idx)
        cv.imwrite(fullfile(dir, file_name), imgs[i, :, :, :])  # faster than PIL or scipy


# compute PSNR
def psnr(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return 10 * math.log10(1 / l2_fun(x, y))


# compute RMSE
def rmse(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return math.sqrt(l2_fun(x, y).item() * 3)  # only works for RGB, for grayscale, don't multiply by 3


# compute SSIM
def ssim(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        return pytorch_ssim.ssim(x, y).item()


# debug, using visdom to visualize images
def vfs(x, padding=10, title=None, ncol=None):
    nrow = 5 if ncol is None else ncol
    t = title if title is not None else ''

    if x.ndim == 3:
        vis.image(x, opts=dict(title=t, caption=t))
    elif x.ndim == 4 and x.shape[0] == 1:
        vis.image(x[0], opts=dict(title=t, caption=t))
    else:
        vis.images(x, opts=dict(title=t, caption=t), nrow=nrow, padding=padding)


def vhm(x, title=None):
    t = title if title is not None else ''
    vis.heatmap(x.squeeze(), opts=dict(title=t, caption=t, layoutopts=dict(plotly=dict(yaxis={'autorange': 'reversed'}))))


# count the number of parameters of a model
def countParameters(model):
    return sum([param.numel() for param in model.parameters() if param.requires_grad])


# generate training title string
def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{}'.format(train_option['data_name'], train_option['model_name'], train_option['loss'],
                                                  train_option['num_train'], train_option['batch_size'], train_option['max_iters'])


# print config
def printConfig():
    print('-------------------------------------- System info -----------------------------------')

    # system
    print('OS: ', platform.platform())  # system build

    # pytorch
    print("torch version=" + torch.__version__)  # PyTorch version
    print("CUDA version=" + torch.version.cuda)  # Corresponding CUDA version
    # print("CUDNN version=" + torch.backends.cudnn.version())  # Corresponding cuDNN version

    # check GPU
    if torch.cuda.device_count() >= 1:
        print('Train with', torch.cuda.device_count(), 'GPUs!')
    else:
        print('Train with CPU!')

    # GPU name
    for i in range(torch.cuda.device_count()):
        print("GPU {:d} name: ".format(i) + torch.cuda.get_device_name(i))  # GPU name

    print('-------------------------------------- System info -----------------------------------')
