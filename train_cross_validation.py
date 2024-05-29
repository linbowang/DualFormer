import os
import datetime
import math
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
from sklearn.model_selection import KFold
import cv2

from test_score import test_score

from DualFormer_831 import DualFormer


# ------------------- Choose Dataset --------------------------
dataset = 'GlaS'
# dataset = 'TNBC'
# dataset = 'MoNuSeg'
# dataset = 'CryoNuSeg'


if dataset == 'GlaS':
    from image_loader_GlaS_5cv import *  # for GlaS
    print('GlaS')
if dataset == 'TNBC':
    print('TNBC')
    from image_loader_TNBC_5cv import *  # for TNBC
if dataset in ['MoNuSeg', 'CryoNuSeg']:
    print('MoNuSeg/CryoNuSeg')
    from image_loader_MoNuSeg_5cv import *  # for MoNuSeg/CryoNuSeg


validation_pred_path = './temp/validation_results/{}'.format(dataset)
validation_gt_path = './temp/validation_gt/{}'.format(dataset)
test_pred_path = './temp/test_results/{}'.format(dataset)
test_gt_path = './temp/test_gt/{}'.format(dataset)


# test temp path
if os.path.exists(test_pred_path):
    shutil.rmtree(test_pred_path)
os.mkdir(test_pred_path)
for i in range(5):
    os.mkdir(test_pred_path + '/{}'.format(i+1))
#
if os.path.exists(test_gt_path):
    shutil.rmtree(test_gt_path)
os.mkdir(test_gt_path)
for i in range(5):
    os.mkdir(test_gt_path + '/{}'.format(i+1))
# validation temp path
if os.path.exists(validation_pred_path):
    shutil.rmtree(validation_pred_path)
os.mkdir(validation_pred_path)
for i in range(5):
    os.mkdir(validation_pred_path + '/{}'.format(i+1))
#
if os.path.exists(validation_gt_path):
    shutil.rmtree(validation_gt_path)
os.mkdir(validation_gt_path)
for i in range(5):
    os.mkdir(validation_gt_path + '/{}'.format(i+1))

data_path = './data/{}'.format(dataset)
training_root = './data/{}/train/'.format(dataset)
test_root = './data/{}/test/'.format(dataset)

cudnn.benchmark = True
torch.cuda.set_device(0)
bce = nn.BCELoss()
bce_logits = nn.BCEWithLogitsLoss().cuda()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def single_gpu_train(batch_size=4, method=None):

    time_now = datetime.datetime.now()
    model_path = './model/{}/{}_{}_{}_{}_{}_{}'.format(dataset, dataset, method, time_now.month, time_now.day, time_now.hour, time_now.minute)
    os.mkdir(model_path)

    if method == 'DualFormer':
        print('DualFormer')
        net = DualFormer().cuda().train()

    # Train Image List
    train_img_name_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(training_root, 'images')) if f.endswith('.jpg') or f.endswith('.png')]
    train_image_list = [(os.path.join(training_root, 'images', img_name + '.png'), os.path.join(training_root, 'masks', img_name + '.png')) for img_name in train_img_name_list]

    # Test Image List
    test_img_name_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(test_root, 'images')) if f.endswith('.jpg') or f.endswith('.png')]
    test_image_list = [(os.path.join(test_root, 'images', img_name + '.png'), os.path.join(test_root, 'masks', img_name + '.png')) for img_name in test_img_name_list]

    test_set = ImageFolder(test_image_list, mode='test',
                           joint_transform=val_joint_transform,
                           img_transform=img_transform,
                           label_transform=label_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
    print('Test Data Loaded')

    # *********************** Cross_validation_iteration *****************************

    kfold = KFold(n_splits=5, shuffle=True, random_state=2024)
    idx = [w for w in range(len(train_image_list))]

    cross_validation_iteration = 0
    for train_idx, validation_idx in kfold.split(idx):
        cross_validation_iteration = cross_validation_iteration + 1
        print('cross_validation_iteration:{}'.format(cross_validation_iteration))
        print('train_idx len:{}'.format(len(train_idx)))
        print('validation_idx len:{}'.format(len(validation_idx)))

        # Reset
        if method == 'DualFormer':
            net = DualFormer().cuda().train()

        optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=5e-4)
        train_list_fold = []
        validation_list_fold = []

        for index in train_idx.tolist():
            train_list_fold.append(train_image_list[index])
        for index in validation_idx.tolist():
            validation_list_fold.append(train_image_list[index])

        train_set = ImageFolder(train_list_fold, mode='train',
                                joint_transform=joint_transform,
                                img_transform=img_transform,
                                label_transform=label_transform)
        validation_set = ImageFolder(validation_list_fold, mode='test',
                                     joint_transform=val_joint_transform,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
        print('Train Data Loaded')
        validation_loader = DataLoader(validation_set, batch_size=1, num_workers=0, shuffle=True, drop_last=True)
        print('Validation Data Loaded')

        max_mdice = 0
        epoch_loss_txt = ''

        for epoch in range(3):
            epoch = epoch
            net.train()
            print('Start epoch[{}/100]'.format(epoch))
            epoch_loss = 0

            # ************************* Train ***************************
            for i, train_data in enumerate(train_loader):
                inputs, labels, _ = train_data

                labels = labels[:, 0, :, :].unsqueeze(1)
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
                optimizer.zero_grad()

                if method == 'DualFormer':
                    out, side_loss = net(inputs, labels)
                    loss = structure_loss(out, labels) + side_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(str(epoch) + ' epoch_loss', epoch_loss)
            epoch_loss_txt = epoch_loss_txt + '{:.4f},'.format(epoch_loss)

            # ********************** Validation **************************
            net.train(False)
            net.eval()

            for i, test_data in enumerate(validation_loader):
                inputs, label, fnames = test_data
                inputs = Variable(inputs).cuda()

                if method == 'DualFormer':
                    pred, _ = net(inputs)
                    pred = pred.squeeze(1).sigmoid().detach().cpu().numpy()

                for j in range(0, len(fnames)):
                    _, cur_fname = os.path.split(fnames[j])
                    cur_fname, _ = os.path.splitext(cur_fname)
                    cur_fname = cur_fname + '.png'
                    gt = label[j][0].squeeze(1).detach().cpu().numpy()
                    gt[gt > 0.5] = 255
                    gt[gt <= 0.5] = 0

                    out = cv2.resize(pred[j], dsize=gt.shape[1::-1])
                    out[out > 0.5] = 255
                    out[out <= 0.5] = 0

                    cv2.imwrite(validation_pred_path + '/{}/{}'.format(cross_validation_iteration, cur_fname), out)
                    cv2.imwrite(validation_gt_path + '/{}/{}'.format(cross_validation_iteration, cur_fname), gt)

            miou, acc, mdice = test_score(validation_pred_path + '/{}'.format(cross_validation_iteration),
                                          validation_gt_path + '/{}'.format(cross_validation_iteration))

            if mdice > max_mdice:
                max_mdice = mdice
                print('saving best_{}.PTH'.format(epoch))
                saved_model_name = '/iteration_{}_best_{}_'.format(cross_validation_iteration, dataset) + method + '.PTH'
                torch.save(net.state_dict(), model_path + saved_model_name)
                with open(model_path + saved_model_name[0:-4] + '_epoch_{}_mdice_{:.2f}_miou_{:.2f}'.format( epoch, mdice * 100, miou * 100) + ".txt", "w") as f:
                    f.write('_')

    print('Start Testing')

    # ************************** Test *****************************
    net = DualFormer().cuda()
    net.train(False)
    net.eval()

    sumDice = 0
    sumIOU = 0
    test_mDice = ''
    test_mIOU = ''
    dice_list = []
    IoU_list = []

    for iter in range(5):
        current_model_path = model_path + '/iteration_{}_best_{}_{}.PTH'.format(iter+1, dataset, method)
        net.load_state_dict(torch.load(current_model_path))

        for i, test_data in enumerate(test_loader):
            inputs, _, fnames = test_data
            inputs = Variable(inputs).cuda()

            pred, _ = net(inputs)
            pred = pred.squeeze(1).sigmoid().detach().cpu().numpy()

            for j in range(0, len(fnames)):
                _, cur_fname = os.path.split(fnames[j])
                cur_fname, _ = os.path.splitext(cur_fname)
                cur_fname = cur_fname + '.png'
                gt = cv2.imread(os.path.join(test_root, 'masks/' + cur_fname))
                out = cv2.resize(pred[j], dsize=gt.shape[1::-1])
                out[out > 0.5] = 255
                out[out <= 0.5] = 0

                cv2.imwrite(test_pred_path + '/{}/'.format(iter + 1) + cur_fname, out)
                cv2.imwrite(test_gt_path + '/{}/'.format(iter + 1) + cur_fname, gt)

        miou, acc, mdice = test_score(test_pred_path + '/{}'.format(iter + 1), test_gt_path + '/{}'.format(iter + 1))
        test_mDice = test_mDice + '_{}_{}'.format(iter, mdice)
        test_mIOU = test_mIOU + '_{}_{}'.format(iter, miou)
        sumDice = sumDice + mdice
        sumIOU = sumIOU + miou
        dice_list.append(mdice * 100)
        IoU_list.append(miou * 100)

    meanDice = sumDice * 100 / 5
    meanIoU = sumIOU * 100 / 5
    stdDice = np.std(np.array(dice_list))
    stdIoU = np.std(np.array(IoU_list))

    with open(model_path + "_test_result.txt", "w") as f:
        # f.write(test_mDice)
        # f.write(test_mIOU)
        f.write('    Dice: {:.2f}±{:.2f}  IoU: {:.2f}±{:.2f}'.format(meanDice, stdDice, meanIoU, stdIoU))


if __name__ == '__main__':
    method = 'DualFormer'

    single_gpu_train(batch_size=4, method=method)
