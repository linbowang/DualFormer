import torch
import warnings
from torch import nn
from torch.nn import functional as F
from dualformer.pvtv2 import pvt_v2_b2
from dualformer.resnet import resnext101_32x8d
from dualformer.utils import cus_sample, upsample_add

from dualformer.segformer2 import M_EfficientSelfAtten, MixFFN

from typing import List, Any, Tuple

bce = nn.BCELoss()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MSCA_2(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSCA_2, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)
        # GAP_branch
        self.GAP_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        # GMP_branch
        self.GMP_att = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.conv7 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        x_GAP = self.GAP_att(x)
        x_GMP = self.GMP_att(x)

        channel_avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        channel_max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        x_channel_pool = torch.cat([channel_avg_pool_out, channel_max_pool_out], dim=1)
        x_channel_pool = self.conv7(x_channel_pool)

        xlg = xl + x_GAP + x_GMP + x_channel_pool
        # xlg = x_GAP + xl
        wei = self.sig(xlg)

        return wei


class ACFM(nn.Module):
    def __init__(self, channel=64):
        super(ACFM, self).__init__()

        # self.msca = MSCA(channels=channel)
        # self.msca2 = MSCA(channels=channel)
        self.msca = MSCA_2(channels=channel)

        self.upsample = cus_sample
        self.conv_x = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv_y = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.convdr = BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv_mask_x = BasicConv2d(channel, 1, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv_mask_y = BasicConv2d(channel, 1, kernel_size=3, stride=1, padding=1, relu=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # -------------------------------
        if x.size(-1) != y.size(-1):
            y = self.upsample(y, scale_factor=2)
        mask_x = self.conv_mask_x(x)
        mask_x = self.sigmoid(mask_x)
        x = self.conv_x(x)
        xo = x * mask_x

        mask_y = self.conv_mask_y(y)
        mask_y = self.sigmoid(mask_y)
        y = self.conv_y(y)
        yo = y * mask_y

        wei = self.msca(xo + yo)
        xo = x * wei + y * (1 - wei)

        # ------------------------------
        # wei2 = self.msca(xo)
        # xo2 = x * wei2 + y * (1 - wei2)

        # ------------------------------
        # ------------------------------
        # if x.size(-1) != y.size(-1):
        #     y = self.upsample(y, scale_factor=2)
        #
        # mask_x = self.conv_mask(x)
        # mask_x = self.sigmoid(mask_x)
        # x = self.conv(x)
        # x = self.relu(x)
        # xo = x * mask_x
        #
        # mask_y = self.conv_mask(y)
        # mask_y = self.sigmoid(mask_y)
        # y = self.conv(y)
        # y = self.relu(y)
        # yo = y * mask_y
        #
        # xo = torch.cat([xo,yo],dim=1)
        # xo = self.convdr(xo)
        # ------------------------------

        return xo


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        CHANNELS_ = [64, 128, 320, 512]

        for i in range(0, len(CHANNELS_)):
            acfm = ACFM(channel=CHANNELS_[i])
            # ca = ChannelAttention(CHANNELS_[i])
            # sa = SpatialAttention()
            setattr(self, f"acfm{i}", acfm)
            # setattr(self, f"ca{i}", ca)
            # setattr(self, f"sa{i}", sa)
        self.convr1 = BasicConv2d(in_planes=256, out_planes=128, kernel_size=1)
        self.convr2 = BasicConv2d(in_planes=512, out_planes=320, kernel_size=1)
        self.convr3 = BasicConv2d(in_planes=1024, out_planes=512, kernel_size=1)

    def forward(self, x, y):
        assert len(x) == len(y)
        fusion_outs = []
        for i in range(0, len(x)):
            acfm = getattr(self, f"acfm{i}")
            # ca = getattr(self, f"ca{i}")
            # sa = getattr(self, f"sa{i}")

            if y[i].size(1) == 256:
                y[i] = self.convr1(y[i])
            elif y[i].size(1) == 512:
                y[i] = self.convr2(y[i])
            elif y[i].size(1) == 1024:
                y[i] = self.convr3(y[i])

            acfm_out = acfm(x[i], y[i])
            fusion_outs.append(acfm_out)

        return fusion_outs


class BridgeLayer_3(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        # self.norm10 = nn.BatchNorm2d(dims*2)
        # self.norm11 = nn.BatchNorm2d(dims*5)
        # self.norm12 = nn.BatchNorm2d(dims*8)
        self.norm10 = nn.LayerNorm(dims * 2)
        self.norm11 = nn.LayerNorm(dims * 5)
        self.norm12 = nn.LayerNorm(dims * 8)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        # self.norm20 = nn.BatchNorm2d(dims * 2)
        # self.norm21 = nn.BatchNorm2d(dims * 5)
        # self.norm22 = nn.BatchNorm2d(dims * 8)
        self.norm20 = nn.LayerNorm(dims * 2)
        self.norm21 = nn.LayerNorm(dims * 5)
        self.norm22 = nn.LayerNorm(dims * 8)

        # self.mixffn1 = MixFFN(dims,dims*4)
        self.mixffn2 = MixFFN(dims * 2, dims * 8)
        self.mixffn3 = MixFFN(dims * 5, dims * 20)
        self.mixffn4 = MixFFN(dims * 8, dims * 32)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        global c2f_N, c3f_N, H, W
        B = inputs[0].shape[0]
        C = 64
        if len(inputs) == 4:
            # print('----4----')
            c1, c2, c3, c4 = inputs
            B, C, H, W = c1.shape
            B, C2, H2, W2 = c2.shape
            B, C3, H3, W3 = c3.shape
            B, C4, H4, W4 = c4.shape

        elif len(inputs) == 3:
            # print('----3----')
            c2, c3, c4 = inputs
            B, C2, H2, W2 = c2.shape
            B, C3, H3, W3 = c3.shape
            B, C4, H4, W4 = c4.shape

        c2_n = self.norm10(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        c3_n = self.norm11(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        c4_n = self.norm12(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # c2fa, c3fa, c4fa = self.attn(self.norm10(c2), self.norm11(c3), self.norm12(c4))
        c2fa, c3fa, c4fa = self.attn(c2_n, c3_n, c4_n)

        c2fa = c2fa.reshape(B, H2, W2, C2).permute(0, 3, 1, 2)
        c3fa = c3fa.reshape(B, H3, W3, C3).permute(0, 3, 1, 2)
        c4fa = c4fa.reshape(B, H4, W4, C4).permute(0, 3, 1, 2)

        tx2 = c2 + c2fa
        tx3 = c3 + c3fa
        tx4 = c4 + c4fa
        tx2 = self.norm20(tx2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        tx3 = self.norm21(tx3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        tx4 = self.norm22(tx4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        tem2 = tx2.reshape(B, -1, C * 2)
        tem3 = tx3.reshape(B, -1, C * 5)
        tem4 = tx4.reshape(B, -1, C * 8)

        # FFNi = EnhancedMix - FFN(split_token)

        m2f = self.mixffn2(tem2, int(H // 2), int(W // 2)).reshape(B, H2, W2, C2).permute(0, 3, 1, 2)
        m3f = self.mixffn3(tem3, int(H // 4), int(W // 4)).reshape(B, H3, W3, C3).permute(0, 3, 1, 2)
        m4f = self.mixffn4(tem4, int(H // 8), int(W // 8)).reshape(B, H4, W4, C4).permute(0, 3, 1, 2)

        res_2, res_3, res_4 = m2f + tx2, m3f + tx3, m4f + tx4

        return [res_2, res_3, res_4]


class BridegeBlock_3(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_3(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_3(dims, head, reduction_ratios)

    def forward(self, x: torch.Tensor) -> List[Any]:
        # x:feature_lists:
        bridge1 = self.bridge_layer1(x)  # [B,3840,64]
        bridge2 = self.bridge_layer2(bridge1)  # [B,3840,64]
        bridge3 = self.bridge_layer3(bridge2)  # [B,3840,64]
        # bridge4 = self.bridge_layer4(bridge2)

        outs = []
        _, C, H0, W0 = x[0].shape
        B, _, H1, W1 = x[1].shape
        _, _, H2, W2 = x[2].shape
        _, _, H3, W3 = x[3].shape

        # sk2 = bridge3[:, :H3 * W3 * 32, :].reshape(B, H3 * 4, W3 * 4, C * 2).permute(0, 3, 1, 2)
        # sk3 = bridge3[:, H3 * W3 * 32:H3 * W3 * 52, :].reshape(B, H3 * 2, H3 * 2, C * 5).permute(0, 3, 1, 2)
        # sk4 = bridge3[:, H3 * W3 * 52:, :].reshape(B, H3, W3, C * 8).permute(0, 3, 1, 2)
        sk2, sk3, sk4 = bridge3

        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)
        return outs


class RFE(nn.Module):
    def __init__(self, channel=512):
        super(RFE, self).__init__()

        dilation_rate = [2, 4, 8, 16]

        self.max_pool = nn.MaxPool2d(3, 2, 1)

        self.conv1 = nn.Sequential(
            BasicConv2d(in_planes=channel, out_planes=1024, kernel_size=3, padding=dilation_rate[0],
                        dilation=dilation_rate[0]),
            BasicConv2d(in_planes=1024, out_planes=256, kernel_size=3, padding=dilation_rate[0],
                        dilation=dilation_rate[0])
        )

        self.conv2 = nn.Sequential(
            BasicConv2d(in_planes=channel, out_planes=1024, kernel_size=3, padding=dilation_rate[1],
                        dilation=dilation_rate[1]),
            BasicConv2d(in_planes=1024, out_planes=256, kernel_size=3, padding=dilation_rate[1],
                        dilation=dilation_rate[1])
        )

        self.conv3 = nn.Sequential(
            BasicConv2d(in_planes=channel, out_planes=1024, kernel_size=3, padding=dilation_rate[2],
                        dilation=dilation_rate[2]),
            BasicConv2d(in_planes=1024, out_planes=256, kernel_size=3, padding=dilation_rate[2],
                        dilation=dilation_rate[2])
        )

        self.conv4 = nn.Sequential(
            BasicConv2d(in_planes=channel, out_planes=1024, kernel_size=3, padding=dilation_rate[3],
                        dilation=dilation_rate[3]),
            BasicConv2d(in_planes=1024, out_planes=256, kernel_size=3, padding=dilation_rate[3],
                        dilation=dilation_rate[3])
        )

        self.conv_out = BasicConv2d(in_planes=256 * 4, out_planes=channel, kernel_size=1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(x)

        conv_cat = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        conv_out = self.conv_out(conv_cat)
        return conv_out


class MFF(nn.Module):
    def __init__(self):
        super(MFF, self).__init__()

        self.conv1 = BasicConv2d(64 * 3, 64, kernel_size=3, padding=1)
        self.last_conv2 = BasicConv2d(in_planes=512, out_planes=64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(128 * 3, 64, kernel_size=3, padding=1)
        self.last_conv3 = BasicConv2d(in_planes=512, out_planes=128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(in_planes=320 * 3, out_planes=128, kernel_size=3, padding=1)
        self.last_conv4 = BasicConv2d(in_planes=512, out_planes=320, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(in_planes=512 * 2, out_planes=320, kernel_size=3, padding=1)

        self.conv4_side = BasicConv2d(in_planes=320, out_planes=1, kernel_size=3, padding=1)
        self.conv3_side = BasicConv2d(in_planes=128, out_planes=1, kernel_size=3, padding=1)
        self.conv2_side = BasicConv2d(in_planes=64, out_planes=1, kernel_size=3, padding=1)

    def forward(self, fist_fusion, in_list, last_fusion, gt=None):
        outs = []
        gt_bce_loss = 0

        last_out = last_fusion

        out4 = torch.cat([in_list[-1], last_out], dim=1)
        outs.append(out4)

        out4 = self.conv4(out4)
        if gt is not None:
            c4_side = nn.Sigmoid()(self.conv4_side(out4))
            gts = F.interpolate(gt, size=(c4_side.size(2), c4_side.size(3)), mode='bilinear', align_corners=True)
            gt_bce_loss += bce(c4_side, gts)
        up_out4 = F.interpolate(out4, scale_factor=2, mode="bilinear")

        last_out4 = self.last_conv4(last_out)
        last_out4 = F.interpolate(last_out4, scale_factor=2, mode="bilinear")

        out3 = torch.cat([in_list[-2], up_out4, last_out4], dim=1)
        outs.append(out3)

        out3 = self.conv3(out3)
        if gt is not None:
            c3_side = nn.Sigmoid()(self.conv3_side(out3))
            gts = F.interpolate(gt, size=(c3_side.size(2), c3_side.size(3)), mode='bilinear', align_corners=True)
            gt_bce_loss += bce(c3_side, gts)
        up_out3 = F.interpolate(out3, scale_factor=2, mode="bilinear")

        last_out3 = self.last_conv3(last_out)
        last_out3 = F.interpolate(last_out3, scale_factor=4, mode="bilinear")

        out2 = torch.cat([in_list[-3], up_out3, last_out3], dim=1)
        outs.append(out2)

        out2 = self.conv2(out2)
        if gt is not None:
            c2_side = nn.Sigmoid()(self.conv2_side(out2))
            gts = F.interpolate(gt, size=(c2_side.size(2), c2_side.size(3)), mode='bilinear', align_corners=True)
            gt_bce_loss += bce(c2_side, gts)
        up_out2 = F.interpolate(out2, scale_factor=2, mode="bilinear")

        last_out2 = self.last_conv2(last_out)
        last_out2 = F.interpolate(last_out2, scale_factor=8, mode="bilinear")

        out1 = torch.cat([fist_fusion, up_out2, last_out2], dim=1)

        outs.append(out1)
        out1 = self.conv1(out1)

        return outs, last_out, out1, gt_bce_loss


class DualFormer(nn.Module):
    def __init__(self, reduction_ratios=None):
        super(DualFormer, self).__init__()

        if reduction_ratios is None:
            self.reduction_ratios = [1, 2, 4]

        self.pvt_backbone = pvt_v2_b2()
        pvt_path = './pretrained_model/pvt_v2_b2.pth'
        pvt_save_model = torch.load(pvt_path)
        pvt_model_dict = self.pvt_backbone.state_dict()
        pvt_state_dict = {k: v for k, v in pvt_save_model.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(pvt_state_dict)
        self.pvt_backbone.load_state_dict(pvt_model_dict)

        self.res_backbone = resnext101_32x8d()
        res_path = './pretrained_model/resnext_101_32x4d.pth'
        res_save_model = torch.load(res_path)
        res_model_dict = self.res_backbone.state_dict()
        res_state_dict = {k: v for k, v in res_save_model.items() if k in res_model_dict.keys()}
        res_model_dict.update(res_state_dict)
        self.res_backbone.load_state_dict(res_model_dict)

        self.fusion = Fusion()

        self.bridge = BridegeBlock_3(64, 1, self.reduction_ratios)

        self.mff = MFF()
        self.last_conv = BasicConv2d(in_planes=512, out_planes=64, kernel_size=1)

        self.pred = nn.Conv2d(64 * 3, 1, kernel_size=1)

    def forward(self, x, gt=None):
        pvt_lists = self.pvt_backbone(x)
        out, res_lists = self.res_backbone(x)
        fusion_lists = self.fusion(pvt_lists, res_lists)
        bridges = self.bridge(fusion_lists)
        outs, last_out, out1, bce_loss = self.mff(fusion_lists[0], bridges, fusion_lists[-1], gt)

        pred = self.pred(outs[-1])
        pred = F.interpolate(pred, scale_factor=4, mode="bilinear")
        out = pred
        return out, bce_loss


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    model = DualFormer()
    pred = model(x)
    print(pred[0].shape)
