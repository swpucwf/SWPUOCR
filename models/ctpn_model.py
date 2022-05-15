import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class basic_conv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 is_relu=True,
                 is_bn=True,
                 is_bias=True):
        '''

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param is_relu:
        :param is_bn:
        :param is_bias:
        '''
        super(basic_conv, self).__init__()

        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=(stride, stride),
                              padding=(padding, padding), dilation=(dilation, dilation), groups=groups, bias=is_bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if is_bn else None
        self.relu = nn.ReLU(inplace=True) if is_relu else None


    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = basic_conv(512, 512, 3, 1, 1, is_bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(256, 512, 1, 1, is_relu=True, is_bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, is_relu=False, is_bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, is_relu=False, is_bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        # rpn

        # 通过rpn 网络
        # torch.Size([1, 512, 14, 14])
        x = self.rpn(x)  # [b, c, h, w]


        #
        x1 = x.permute(0, 2, 3, 1).contiguous()  # channels last   [b, h, w, c]
        b = x1.size()  # b, h, w, c
        x1 = x1.view(b[0] * b[1], b[2], b[3])
        # 通过双向LSTM
        x2, _ = self.brnn(x1)

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4, 20, 20, 256])

        x3 = x3.permute(0, 3, 1, 2).contiguous()  # channels first [b, c, h, w]
        # 输出
        # print(x3.shape)
        # torch.Size([1, 256, 14, 14])
        x3 = self.lstm_fc(x3)
        # torch.Size([1, 512, 14, 14])
        # print(x3.shape)

        x = x3

        # [1, 20, 14, 14]
        cls = self.rpn_class(x)
        # print(cls.shape)
        # torch.Size([1, 20, 14, 14])
        regr = self.rpn_regress(x)

        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()

        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)

        return cls, regr


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = CTPN_Model()
    print(model)
    cls, regr = model(x)
    print(cls.shape)
    print(regr.shape)
