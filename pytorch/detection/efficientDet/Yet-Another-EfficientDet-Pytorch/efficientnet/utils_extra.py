# Author: Zylo117

import math

from torch import nn
import torch.nn.functional as F


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        if type(stride) == int:
            inp_stride = [stride, stride]
        elif len(stride) == 1:
            inp_stride = [stride[0], stride[0]]
        else:
            inp_stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=inp_stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2
        self.first_time = True

    def forward(self, x):
        if self.first_time:
            h, w = x.shape[-2:]
            
            extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
            extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
            
            left = extra_h // 2
            right = extra_h - left
            top = extra_v // 2
            bottom = extra_v - top
            if left > 0 or right > 0 or top > 0 or bottom > 0:
                self.static_padding = nn.ZeroPad2d((left, right, top, bottom))
            else:
                self.static_padding = Identity()
            self.first_time = False

        x = self.static_padding(x)

        x = self.conv(x)
        return x

# class Conv2dStaticSamePadding(nn.Conv2d):
#     """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
#        The padding mudule is calculated in construction function, then used in forward.
#     """

#     # With the same calculation as Conv2dDynamicSamePadding

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
#         super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
#         self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

#         # Calculate padding based on image size and save it
#         assert image_size is not None
#         ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
#         kh, kw = self.weight.size()[-2:]
#         sh, sw = self.stride
#         oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
#         pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
#         pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
#         if pad_h > 0 or pad_w > 0:
#             self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
#         else:
#             self.static_padding = Identity()
#         print((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

#     def forward(self, x):
#         x = self.static_padding(x)
#         x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return x

# class MaxPool2dStaticSamePadding(nn.MaxPool2d):
#     """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
#        The padding mudule is calculated in construction function, then used in forward.
#     """

#     def __init__(self, kernel_size, stride, image_size=None, **kwargs):
#         super().__init__(kernel_size, stride, **kwargs)
#         self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
#         self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
#         self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

#         # Calculate padding based on image size and save it
#         assert image_size is not None
#         ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
#         kh, kw = self.kernel_size
#         sh, sw = self.stride
#         oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
#         pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
#         pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
#         if pad_h > 0 or pad_w > 0:
#             self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
#         else:
#             self.static_padding = Identity()

#     def forward(self, x):
#         x = self.static_padding(x)
#         x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
#                          self.dilation, self.ceil_mode, self.return_indices)
#         return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2
        self.first_time = True

    def forward(self, x):
        if self.first_time:
            h, w = x.shape[-2:]
            
            extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
            extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

            left = extra_h // 2
            right = extra_h - left
            top = extra_v // 2
            bottom = extra_v - top
            if left > 0 or right > 0 or top > 0 or bottom > 0:
                self.static_padding = nn.ZeroPad2d((left, right, top, bottom))
            else:
                self.static_padding = Identity()
            self.first_time = False

        x = self.static_padding(x)

        x = self.pool(x)
        return x


class Identity(nn.Module):
    """Identity mapping.
       Send input to output directly.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

