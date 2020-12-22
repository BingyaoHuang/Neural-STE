"""
Training and testing script for Neural-STE (AAAI'21)

This script trains/tests Neural-STE on different setups specified in 'data_list' below.
The detailed training options are given in 'train_option' below.
1. We start by setting the training environment to GPU (if any).
2. Real and synthetic setups are listed in 'data_list', more setups can be found in ../../data folder.
3. We set number of training images to 450 and loss function to l1+ssim, you can add other num_train and loss to 'num_train_list' and 'loss_list' for
comparison. Other training options are specified in 'train_option'.
4. The training data 'train_data' and validation data 'valid_data', are loaded in RAM using function 'loadData', and then we train the model with
function 'trainNeuralSTE'. The training and validation results are both updated in Visdom window (`http://server:8098`) and console.
5. Once the training is finished, the testing results are saved to [dataset_root]/[data_name]/pred/.

Example:
    python train_Neural-STE.py

See Models.py for Neural-STE structure.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.


Citation:
    @inproceedings{huang2021Neural-STE,
        title={Modeling Deep Learning Based Privacy Attacks on Physical Mail},
        author={Bingyao Huang and Ruyi Lian and Dimitris Samaras and Haibin Ling},
        booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
        year={2021}}
"""

# %% Set environment
import os

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device_ids = [0, 1, 2]

from trainNetwork import *
import Models
from time import localtime, strftime

printConfig()

# set PyTorch device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For final results, set this to true
reproducible = True

# reproducibility
if reproducible:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

dataset_root = fullfile(os.getcwd(), '../../data')

# all data
data_list = [
    # # real attacks
    # 'real/easy',
    # 'real/medium',
    'real/hard',

    # real countering attacks
    'real/counter_unsafe',
    'real/counter_safe',

    # synthetic attacks, naming: synth_hx_L_kA_kt
    'synthetic/synth_3_0.6_0.8_0.5',
    'synthetic/synth_6_0.4_0.8_0.5',
    'synthetic/synth_6_0.6_0.6_0.5',
    'synthetic/synth_6_0.6_0.8_0.1',
    'synthetic/synth_6_0.6_0.8_0.3',
    'synthetic/synth_6_0.6_0.8_0.5',
    'synthetic/synth_6_0.6_0.8_0.7',
    'synthetic/synth_6_0.6_1.0_0.5',
    'synthetic/synth_6_0.6_1.0_1.0',
    'synthetic/synth_6_0.8_0.8_0.5',
    'synthetic/synth_9_0.4_0.4_0.3',
    'synthetic/synth_9_0.6_0.8_0.5',

    # synthetic countering attacks, a safe envelope: hx=17, L=0.6, kA=1.0, kt=0.1
    'synthetic/synth_17_0.6_1.0_0.1',
]

num_train_list = [450]

# degradation_list = ['', 'black_box', 'no_warp', 'no_refine', 'no_A_const', 'no_J_const'] # a list of degraded versions
degradation_list = ['']  # The proposed method w/o any degradation

# default train options
train_option_default = {'data_name': '',  # will be set later
                        'model_name': '',
                        'num_train': 450,  # used to test different number of training samples
                        'max_iters': 4000,
                        'batch_size': 16,
                        'lr': [1e-3, 1e-3],  # learning rate
                        'lr_drop_ratio': 0.2,
                        'lr_drop_rate': [3400, 3800],
                        'loss': 'l1+ssim',
                        'l2_reg': [0, 5e-4],  # l2 regularization. 5e-4 is better with
                        'device': device,
                        'device_ids': device_ids,
                        'plot_on': True,  # plot training progress using visdom, disable this for faster training
                        'train_plot_rate': 50,  # training and visdom plot rate, use a larger rate for faster training
                        'valid_rate': 200}  # validation and visdom plot rate, use a larger rate for faster training

save_pred = True  # save model recovered hidden content to files
input_size = None  # original size, no resize
gray_scale = False  # set to True when compare with PSDNet

# log file
log_dir = '../../log'
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file_name = strftime('%Y-%m-%d_%H_%M_%S', localtime()) + '.txt'
log_file = open(fullfile(log_dir, log_file_name), 'w')
title_str = '{:40s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('data_name', 'model_name', 'num_train', 'batch_size', 'loss_function',
                                'cap_psnr', 'cap_rmse', 'cap_ssim',  # cam-captured
                                'valid_psnr', 'valid_rmse', 'valid_ssim', 'max_iters'))  # inferred hidden J
log_file.close()

# start training for all data
for data_name in data_list:
    data_root = fullfile(dataset_root, data_name)

    cam_train, cam_valid, gt_train, gt_valid = loadData(dataset_root, data_name, input_size, gray_scale=gray_scale)

    # stats for different #Train
    for num_train in num_train_list:
        train_option = train_option_default.copy()
        train_option['num_train'] = num_train
        train_option['data_name'] = data_name.replace('/', '_')

        if gray_scale:
            train_option['lr'] = [1e-2, 2e-4]
            train_option['l2_reg'] = [1e-4, 5e-4]

        # training and validation data (I: cam-captured envelope; J: hidden content GT)
        train_data = {'I': cam_train[:num_train, :, :, :], 'J': gt_train[:num_train, :, :, :]}
        valid_data = {'I': cam_valid, 'J': gt_valid}

        for degradation in degradation_list:
            # for repeatability
            resetRNGseed(0)

            # WarpingNet for geometric correction
            warping_net = Models.WarpingNet(chan_in=train_data['I'].shape[1], out_size=gt_train.shape[2:4])
            if torch.cuda.device_count() >= 1: warping_net = nn.DataParallel(warping_net, device_ids=device_ids).to(device)

            # Dehazing and RefineNet
            dehazing_refine_net = Models.DehazingRefineNet(chan_in=train_data['I'].shape[1], chan_out=train_data['J'].shape[1], degradation=degradation)
            if torch.cuda.device_count() >= 1: dehazing_refine_net = nn.DataParallel(dehazing_refine_net, device_ids=device_ids).to(device)

            # Neural-STE
            model = Models.NeuralSTE(warping_net, dehazing_refine_net, degradation=degradation)  # with GAN
            if torch.cuda.device_count() >= 1: model = nn.DataParallel(model, device_ids=device_ids).to(device)

            # train
            print('-------------------------------------- Training Options -----------------------------------')
            print("\n".join("{}: {}".format(k, v) for k, v in train_option.items()))
            print('-------------------------------------- Start training {:s} ---------------------------'.format(model.module.name))

            model, valid_psnr, valid_rmse, valid_ssim = trainNeuralSTE(model, train_data, valid_data, train_option)

            # save model predictions
            if save_pred:
                model_config = '{}_{}_{}'.format(model.module.name, num_train, train_option['max_iters'])
                print('------------------------------------ Save testing results for {:s} ---------------------------'.format(model.module.name))

                # create results folder
                pred_path = fullfile(data_root, 'pred/test', model_config)
                if not os.path.exists(pred_path): os.makedirs(pred_path)

                # save to images
                with torch.no_grad():
                    pred_tuple = model(cam_valid.to(device))
                    saveImgs(pred_tuple[0].detach().cpu(), fullfile(pred_path, 'J_hat'))
                    saveImgs(pred_tuple[1].detach().cpu(), fullfile(pred_path, 'I_warp'))
                    saveImgs(pred_tuple[2].detach().cpu(), fullfile(pred_path, 'J_coarse'))
                    saveImgs(pred_tuple[3].detach().cpu(), fullfile(pred_path, 'A'))
                    saveImgs(pred_tuple[4].detach().cpu(), fullfile(pred_path, 't'))
                print('Images saved to ' + pred_path)

            # save results to log
            ret_str = '{:40s}{:<30}{:<20}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15}\n'

            if not cam_valid.shape == gt_valid.shape:
                cam_valid_resize = F.interpolate(cam_valid, gt_valid.shape[2:4])
            else:
                cam_valid_resize = cam_valid

            log_file = open(fullfile(log_dir, log_file_name), 'a')
            log_file.write(
                ret_str.format(data_name, model.module.name, num_train, train_option['batch_size'], train_option['loss'],
                               psnr(cam_valid_resize, gt_valid), rmse(cam_valid_resize, gt_valid), ssim(cam_valid_resize, gt_valid),
                               valid_psnr, valid_rmse, valid_ssim, train_option['max_iters']))
            log_file.close()

print("Done training")
