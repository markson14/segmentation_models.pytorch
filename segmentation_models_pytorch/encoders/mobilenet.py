""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torchvision
import torch.nn as nn
import torch.nn.functional as F
from ._base import EncoderMixin


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl="RE"):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == "RE":
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == "HS":
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride,
                       padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode="small", width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == "large":
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, "RE", 1],
                [3, 64,  24,  False, "RE", 2],
                [3, 72,  24,  False, "RE", 1],
                [5, 72,  40,  True,  "RE", 2],
                [5, 120, 40,  True,  "RE", 1],
                [5, 120, 40,  True,  "RE", 1],
                [3, 240, 80,  False, "HS", 2],
                [3, 200, 80,  False, "HS", 1],
                [3, 184, 80,  False, "HS", 1],
                [3, 184, 80,  False, "HS", 1],
                [3, 480, 112, True,  "HS", 1],
                [3, 672, 112, True,  "HS", 1],
                [5, 672, 160, True,  "HS", 2],
                [5, 960, 160, True,  "HS", 1],
                [5, 960, 160, True,  "HS", 1],
            ]
        elif mode == "small":
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  "RE", 2],
                [3, 72,  24,  False, "RE", 2],
                [3, 88,  24,  False, "RE", 1],
                [5, 96,  40,  True,  "HS", 2],
                [5, 240, 40,  True,  "HS", 1],
                [5, 240, 40,  True,  "HS", 1],
                [5, 120, 48,  True,  "HS", 1],
                [5, 144, 48,  True,  "HS", 1],
                [5, 288, 96,  True,  "HS", 2],
                [5, 576, 96,  True,  "HS", 1],
                [5, 576, 96,  True,  "HS", 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(
                input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == "large":
            last_conv = make_divisible(960 * width_mult)
        elif mode == "small":
            last_conv = make_divisible(576 * width_mult)
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        else:
            raise NotImplementedError
        # classifier head
        self.features.append(conv_1x1_bn(
            input_channel, last_conv, nlin_layer=Hswish))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        self.features.append(Hswish(inplace=True))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class MobileNetV3Encoder(MobileNetV3, EncoderMixin):
    def __init__(self, depth=5, mode="small", width_mult=1.0, **kwargs):
        super().__init__(mode=mode, width_mult=width_mult, **kwargs)
        self._depth = depth
        self.mode = mode
        if mode == "small":
            self._out_channels = (3*width_mult, 16*width_mult, 16*width_mult,
                                  24*width_mult, 48*width_mult, 96*width_mult)
            self._out_channels = tuple(map(make_divisible, self._out_channels))
        elif mode == "large":
            self._out_channels = (3*width_mult, 16*width_mult, 24*width_mult,
                                  40*width_mult, 112*width_mult, 160*width_mult)
            self._out_channels = tuple(map(make_divisible, self._out_channels))
        else:
            raise NameError
        self._in_channels = 3

        del self.classifier

    def get_stages(self):
        if self.mode == "small":
            return [
                nn.Identity(),
                self.features[:1],
                self.features[1:2],
                self.features[2:4],
                self.features[4:9],
                self.features[9:-4],
            ]
        elif self.mode == "large":
            return [
                nn.Identity(),
                self.features[:2],
                self.features[2:4],
                self.features[4:7],
                self.features[7:13],
                self.features[13:-4],
            ]
        else:
            raise NotImplementedError

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features


class MobileNetV2Encoder(torchvision.models.MobileNetV2, EncoderMixin):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias")
        state_dict.pop("classifier.1.weight")
        super().load_state_dict(state_dict, **kwargs)


mobilenet_encoders = {
    "mobilenet_v2": {
        "encoder": MobileNetV2Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 24, 32, 96, 1280),
        },
    },
    "mobilenetv3_s_1.0": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "mode": "small",
            "width_mult": 1.0,
        }
    },
    "mobilenetv3_s_0.25": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "mode": "small",
            "width_mult": 0.25,
        }
    },
    "mobilenetv3_l_1.0": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "mode": "large",
            "width_mult": 1.0,
        }
    }
}
