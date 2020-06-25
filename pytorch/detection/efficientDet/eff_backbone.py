# Author: Zylo117

import torch
import sys
sys.path.append("./Yet-Another-EfficientDet-Pytorch")
import math
import torch
from torch import nn
from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors, transform_anchors


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, onnx_export=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False, onnx_export=onnx_export,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef], onnx_export=onnx_export)
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef], onnx_export=onnx_export)

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)
        print("Input size is {}".format(self.input_sizes[self.compound_coef]))
        self.forward_anchors = self.anchors((self.input_sizes[self.compound_coef], self.input_sizes[self.compound_coef]))
        self.anchor_data = transform_anchors(self.forward_anchors) / self.input_sizes[self.compound_coef]
        self.total_anchors = self.anchor_data.shape[0]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = (torch.ones(1,1) * inputs.shape[-1]).data

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)[0, :, :]
        classification = self.classifier(features)[0, :, :]
        # box_y = ((regression[:, 0:1] * self.anchor_data[2]) + self.anchor_data[0]) / max_size
        # box_x = ((regression[:, 1:2] * self.anchor_data[3]) + self.anchor_data[1]) / max_size
        # box_h = (regression[:, 2:3].exp() * self.anchor_data[2]) / max_size
        # box_w = (regression[:, 3:4].exp() * self.anchor_data[3]) / max_size

        # box_y = torch.clamp(box_y, min=0, max=1)
        # box_x = torch.clamp(box_x, min=0, max=1)
        # box_h = torch.clamp(box_h, min=0, max=1)
        # box_w = torch.clamp(box_w, min=0, max=1)
        # final_boxes = torch.cat((box_x, box_y, box_w, box_h), axis=1)

        # return classification 
        
        return regression, classification

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')