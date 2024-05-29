import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
import cv2
from test_score import test_score

from image_loader_PanNuke import *

from DualFormer_831 import DualFormer

dataset = 'PanNuke'

model_path = './model/{}'.format(dataset)
data_path = './data/{}'.format(dataset)

pred_path = './results/{}'.format(dataset)
gt_path = './data/{}/fold1/masks'.format(dataset)

training_root = './data/{}/fold3'.format(dataset)
test_root = './data/{}/fold1'.format(dataset)

if not os.path.exists(pred_path):
    os.mkdir(pred_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)
# -----------------------------------------------
# train
cudnn.benchmark = True
torch.cuda.set_device(0)
bce_logits = nn.BCEWithLogitsLoss().cuda()
bce = nn.BCELoss()
criterion_MSE = torch.nn.MSELoss().cuda()


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

    if method == 'DualFormer':
        print('DualFormer')
        net = DualFormer().cuda().train()

    # Datasets
    optimizer = torch.optim.Adamax([{'params': net.parameters()}], lr=5e-4)

    train_set = ImageFolder(training_root, mode='train',
                            joint_transform=joint_transform,
                            img_transform=img_transform,
                            label_transform=label_transform)
    test_set = ImageFolder(test_root, mode='test',
                           joint_transform=val_joint_transform,
                           img_transform=img_transform,
                           label_transform=label_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    print(len(train_loader))
    print('Train Data Loaded')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
    print(len(test_loader))
    print('Test Data Loaded')
    max_mdice = 0

    epoch_loss_txt = ''
    epoch_pred_loss_txt = ''
    epoch_dice_txt = ''
    epoch_miou_txt = ''

    for epoch in range(40):
        epoch = epoch
        print('Start epoch[{}/40]'.format(epoch))
        epoch_loss = 0
        epoch_pred_loss = 0
        net.train()
        for i, train_data in list(enumerate(train_loader)):
            inputs, labels, _ = train_data
            labels = labels[:, 0, :, :].unsqueeze(1)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()

            if method == 'DualFormer':
                out, side_loss = net(inputs, labels)
                pred_loss = structure_loss(out, labels)
                loss = pred_loss + side_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_pred_loss += pred_loss.item()

        print(str(epoch) + ' epoch_loss', epoch_loss)
        print(str(epoch) + ' epoch_pred_loss', epoch_pred_loss)

        # logs
        epoch_loss_txt = epoch_loss_txt + '{:.4f},'.format(epoch_loss)
        with open(model_path + "/epoch_loss.txt", "w") as f:
            f.write(epoch_loss_txt)
        epoch_pred_loss_txt = epoch_pred_loss_txt + '{:.4f},'.format(epoch_pred_loss)
        with open(model_path + "/epoch_pred_loss.txt", "w") as f:
            f.write(epoch_pred_loss_txt)

        # ******************************************* TEST ************************************************
        net.train(False)
        net.eval()

        for i, test_data in enumerate(test_loader):
            inputs, _, fnames = test_data
            inputs = Variable(inputs).cuda()

            if method == 'DualFormer':
                pred, _ = net(inputs)
                pred = pred.sigmoid().squeeze(1).detach().cpu().numpy()

            for j in range(0, len(fnames)):
                _, cur_fname = os.path.split(fnames[j])
                cur_fname, _ = os.path.splitext(cur_fname)
                cur_fname = cur_fname + '.png'
                gt = cv2.imread(os.path.join(test_root, 'masks/' + cur_fname))
                out = cv2.resize(pred[j], dsize=gt.shape[1::-1])
                out[out > 0.5] = 255
                out[out <= 0.5] = 0
                cv2.imwrite(os.path.join(pred_path, cur_fname), out)

        miou, acc, mdice = test_score(gt_path, pred_path)

        # logs
        epoch_dice_txt = epoch_dice_txt + '{:.2f},'.format(mdice * 100)
        with open(model_path + "/epoch_dice.txt", "w") as f:
            f.write(epoch_dice_txt)
        epoch_miou_txt = epoch_miou_txt + '{:.2f},'.format(miou * 100)
        with open(model_path + "/epoch_miou.txt", "w") as f:
            f.write(epoch_miou_txt)

        if mdice > max_mdice:
            max_mdice = mdice
            print('saving best_{}.PTH'.format(epoch))
            saved_model_path = model_path + '/best_{}_epoch_{}_dice_{:.2f}_IoU_{:.2f}.PTH'.format(dataset, epoch, mdice * 100, miou * 100)
            torch.save(net.state_dict(), saved_model_path)


if __name__ == '__main__':

    method = 'DualFormer'

    single_gpu_train(batch_size=4, method=method)
