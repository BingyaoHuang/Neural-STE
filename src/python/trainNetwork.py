'''
Neural-STE training functions
'''

from utils import *
import torch.nn.functional as F
import torch.optim as optim
import time

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM().cuda()


# %% load Neural-STE training and validation data
def loadData(dataset_root, data_name, input_size, gray_scale=False):
    # data paths
    data_root = fullfile(dataset_root, data_name)
    cam_train_path = fullfile(data_root, 'train')
    cam_valid_path = fullfile(data_root, 'test')
    gt_train_path = fullfile(dataset_root, 'gt/train')
    gt_valid_path = fullfile(dataset_root, 'gt/test')
    print("Loading data from '{}'".format(data_root))

    # training data
    cam_train = readImgsMT(cam_train_path, size=input_size, gray_scale=gray_scale)
    gt_train = readImgsMT(gt_train_path, size=input_size, gray_scale=gray_scale)

    # validation data
    cam_valid = readImgsMT(cam_valid_path, size=input_size, gray_scale=gray_scale)
    gt_valid = readImgsMT(gt_valid_path, size=input_size, gray_scale=gray_scale)

    return cam_train, cam_valid, gt_train, gt_valid


# %% train Neural-STE (all images are loaded to RAM)
def trainNeuralSTE(model, train_data, valid_data, train_option):
    device = train_option['device']

    # empty cuda cache before training
    if device.type == 'cuda': torch.cuda.empty_cache()

    # training data
    I = train_data['I']
    J = train_data['J']

    # list of all parameter to be optimized, including net parameters
    warping_params = list(map(lambda x: x[1], list(filter(lambda kv: 'module.warping_net' in kv[0], model.named_parameters()))))  # WarpingNet
    other_params = list(map(lambda x: x[1], list(filter(lambda kv: 'module.dehazing_refine_net' in kv[0], model.named_parameters()))))  # DehazingRefineNet

    # optimizers
    w_optimizer = optim.Adam([{'params': warping_params}], lr=train_option['lr'][0], weight_decay=train_option['l2_reg'][0])  # WarpingNet
    o_optimizer = optim.Adam([{'params': other_params}], lr=train_option['lr'][1], weight_decay=train_option['l2_reg'][1])  # DehazingRefineNet

    w_lr_scheduler = optim.lr_scheduler.MultiStepLR(w_optimizer, milestones=train_option['lr_drop_rate'], gamma=train_option['lr_drop_ratio'])
    o_lr_scheduler = optim.lr_scheduler.MultiStepLR(o_optimizer, milestones=train_option['lr_drop_rate'], gamma=train_option['lr_drop_ratio'])

    # %% start train
    start_time = time.time()

    # get model name
    train_option['model_name'] = model.name if hasattr(model, 'name') else model.module.name

    # initialize visdom data visualization figure
    if 'plot_on' not in train_option: train_option['plot_on'] = True

    # title string of current training option
    title = optionToString(train_option)
    if train_option['plot_on']:
        # intialize visdom figures
        vis_train_fig = None
        vis_valid_fig = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]), name='origin',
                                 opts=dict(width=1300, height=500, markers=True, markersize=3,
                                           layoutopts=dict(
                                               plotly=dict(title={'text': title, 'font': {'size': 24}},
                                                           font={'family': 'Arial', 'size': 20},
                                                           hoverlabel={'font': {'size': 20}},
                                                           xaxis={'title': 'Iteration'},
                                                           yaxis={'title': 'Metrics', 'hoverformat': '.4f'}))))

    # main loop
    iters = 0

    while iters < train_option['max_iters']:
        model.train()  # explicitly set to train mode in case batchNormalization and dropout are used

        # randomly sample training batch and send to GPU
        idx = random.sample(range(train_option['num_train']), train_option['batch_size'])
        I_batch = I[idx, :, :, :].to(device) if I.device.type != 'cuda' else I[idx, :, :, :]  # I
        J_batch = J[idx, :, :, :].to(device) if J.device.type != 'cuda' else J[idx, :, :, :]  # J

        # predict and compute loss
        J_hat, I_warp, J_coarse, A, t = model(I_batch)  # we omit _batch suffix for model outputs
        train_loss_batch, train_l2_loss_batch = computeLoss(J_hat, J_batch, train_option['loss'])

        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channel, rgb

        if model.module.name == 'Neural-STE':
            # loss used in paper
            if 'no_J_const' not in model.module.name:
                train_loss_batch += 1 * l2_fun(J_coarse, J_batch)  # J_coarse should also be similar to GT (J_batch)
            if 'no_A_const' not in model.module.name:
                train_loss_batch += 0.1 * l2_fun(A, I_warp)  # A should be like warped image I_warp=T(I)

        # backpropagation and params update
        w_optimizer.zero_grad()
        o_optimizer.zero_grad()
        train_loss_batch.backward()
        w_optimizer.step()
        o_optimizer.step()

        # record running time
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

        # plot train
        if train_option['plot_on']:
            if iters % train_option['train_plot_rate'] == 0 or iters == train_option['max_iters'] - 1:
                vis_train_fig = plotMontageMultirow(I_batch, J_hat, J_batch, win=vis_train_fig, title='[Train]' + title, save=True, iter=iters, transpose=False)
                appendDataPoint(iters, train_loss_batch.item(), vis_curve_fig, 'train_loss')
                appendDataPoint(iters, train_rmse_batch, vis_curve_fig, 'train_rmse')

        # validation
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        if valid_data is not None and (iters % train_option['valid_rate'] == 0 or iters == train_option['max_iters'] - 1):
            valid_psnr, valid_rmse, valid_ssim, J_valid, I_warp_valid, J_coarse_valid, A_valid, t_valid = evalNeuralSTE(model, valid_data)

            # plot validation
            if train_option['plot_on']:
                idx = [0, 6, 24, 28, 33]
                vis_valid_fig = plotMontageMultirow(valid_data['I'][idx], J_valid[idx], valid_data['J'][idx], win=vis_valid_fig, title='[Valid]' + title, save=True, iter=iters, transpose=False)
                appendDataPoint(iters, valid_rmse, vis_curve_fig, 'valid_rmse')
                appendDataPoint(iters, valid_ssim, vis_curve_fig, 'valid_ssim')

        # print results to console
        print('Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Valid PSNR: {:7s}  | Valid RMSE: {:6s}  '
              '| Valid SSIM: {:6s}  | Learn Rate: {:.5f}/{:.5f} |'.format(iters, time_lapse, train_loss_batch.item(), train_rmse_batch,
                                                                          '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                          '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                          '{:.4f}'.format(valid_ssim) if valid_ssim else '',
                                                                          w_optimizer.param_groups[0]['lr'],
                                                                          o_optimizer.param_groups[0]['lr']))

        w_lr_scheduler.step()
        o_lr_scheduler.step()

        iters += 1

    # Done training and save the last epoch model
    checkpoint_dir = '../../checkpoint'
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    checkpoint_file_name = fullfile(checkpoint_dir, title + '.pth')
    torch.save(model.state_dict(), checkpoint_file_name)
    print('Checkpoint saved to {}\n'.format(checkpoint_file_name))

    return model, valid_psnr, valid_rmse, valid_ssim


# %% local functions

# compute loss between prediction and ground truth
def computeLoss(prj_pred, prj_train, loss_option):
    train_loss = 0

    # l1
    if 'l1' in loss_option:
        l1_loss = l1_fun(prj_pred, prj_train)
        train_loss += l1_loss

    # l2
    l2_loss = l2_fun(prj_pred, prj_train)
    if 'l2' in loss_option:
        train_loss += l2_loss

    # ssim
    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))
        train_loss += ssim_loss

    return train_loss, l2_loss


# append a data point to the curve in win
def appendDataPoint(x, y, win, name, env=None):
    vis.line(
        X=np.array([x]),
        Y=np.array([y]),
        env=env,
        win=win,
        update='append',
        name=name,
        opts=dict(markers=True, markersize=3)
    )


# plot sample predicted images using visdom, more than three rows
def plotMontageMultirow(*argv, index=None, win=None, title=None, env=None, normalize=False, save=False, iter=None, transpose=True):
    with torch.no_grad():  # just in case
        # compute montage grid size
        if argv[0].shape[0] > 5:
            grid_w = 5
            idx = random.sample(range(argv[0].shape[0]), grid_w) if index is None else index
        else:
            grid_w = argv[0].shape[0]
            # idx = random.sample(range(cam_im.shape[0]), grid_w)
            idx = range(grid_w)

        # resize to (256, 256) for better display
        tile_size = (256, 256)
        im_resize = torch.empty((len(argv) * grid_w, argv[0].shape[1]) + tile_size)
        i = 0
        for im in argv:
            if im.shape[2] != tile_size[0] or im.shape[3] != tile_size[1]:
                im_resize[i:i + grid_w] = F.interpolate(im[idx, :, :, :], tile_size)
            else:
                im_resize[i:i + grid_w] = im[idx, :, :, :]
            i += grid_w

        # title
        plot_opts = dict(title=title, caption=title, font=dict(size=18), width=1300, store_history=False)

        # unnormalize to [0, 1] if image is normalized to [-1, 1]
        if normalize:
            im_resize = im_resize * 0.5 + 0.5

        # plot montage to existing win, otherwise create a new win
        if transpose:
            im_montage = make_grid_transposed(im_resize, nrow=grid_w, padding=10, pad_value=1)
        else:
            im_montage = torchvision.utils.make_grid(im_resize, nrow=grid_w, padding=10, pad_value=1)

        win = vis.image(im_montage, win=win, opts=plot_opts, env=env)

        # save to files for gif
        if save:
            save_path = fullfile('../../gif', title)
            if not os.path.exists(save_path): os.makedirs(save_path)
            torchvision.utils.save_image(im_montage, fullfile(save_path, 'iter_{:04d}.png'.format(iter)))
    return win


# evaluate Neural-STE using validation data
def evalNeuralSTE(model, valid_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    I = valid_data['I']
    J = valid_data['J']

    with torch.no_grad():
        model.eval()  # explicitly set to eval mode

        # for limited GPU memory, we need to perform validation in batch mode
        last_loc = 0
        valid_mse, valid_ssim = 0., 0.

        J_hat = torch.zeros(J.shape)
        I_warp = torch.zeros(J.shape)
        A = torch.zeros(J.shape)
        t = torch.zeros(J.shape)
        J_coarse = torch.zeros(J.shape)
        num_valid = I.shape[0]
        batch_size = 50 if num_valid > 50 else num_valid  # default number of test images per dataset

        for i in range(0, num_valid // batch_size):
            idx = range(last_loc, last_loc + batch_size)
            I_batch = I[idx, :, :, :].to(device) if I.device.type != 'cuda' else I[idx, :, :, :]
            J_batch = J[idx, :, :, :].to(device) if J.device.type != 'cuda' else J[idx, :, :, :]

            # predict batch
            J_hat_batch = model(I_batch)
            if type(J_hat_batch) == tuple and len(J_hat_batch) > 1:
                I_warp_batch = J_hat_batch[1]
                J_coarse_batch = J_hat_batch[2]
                A_batch = J_hat_batch[3]
                t_batch = J_hat_batch[4]
                J_hat_batch = J_hat_batch[0]
            J_hat_batch = J_hat_batch.detach()
            I_warp_batch = I_warp_batch.detach()
            J_coarse_batch = J_coarse_batch.detach()
            A_batch = A_batch.detach()
            t_batch = t_batch.detach()

            if type(J_hat_batch) == tuple and len(J_hat_batch) > 1: J_hat_batch = J_hat_batch[0]
            J_hat[last_loc:last_loc + batch_size, :, :, :] = J_hat_batch.cpu()
            I_warp[last_loc:last_loc + batch_size, :, :, :] = I_warp_batch.cpu()
            J_coarse[last_loc:last_loc + batch_size, :, :, :] = J_coarse_batch.cpu()
            A[last_loc:last_loc + batch_size, :, :, :] = A_batch.cpu()
            t[last_loc:last_loc + batch_size, :, :, :] = t_batch.cpu()

            # compute loss
            valid_mse += l2_fun(J_hat_batch, J_batch).item() * batch_size
            valid_ssim += ssim(J_hat_batch, J_batch) * batch_size

            last_loc += batch_size
        # average
        valid_mse /= num_valid
        valid_ssim /= num_valid

        valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
        valid_psnr = 10 * math.log10(1 / valid_mse)

    return valid_psnr, valid_rmse, valid_ssim, J_hat, I_warp, J_coarse, A, t
