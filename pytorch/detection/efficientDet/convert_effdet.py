import torch
import yaml
from torch import nn
from eff_backbone import EfficientDetBackbone
import numpy as np
import onnx_coreml
import os
from utils import *
import onnx
import argparse
import coremltools
from coremltools.models import datatypes
from coremltools.models.pipeline import *

#at the time of writing this code coreml and onnx_coreml had some bugs in source code that needs to changed
#for the conversion process to proceed
#add the tem "rank=4" in line 46 and 337 at onnx_coreml/_operators_nd.py
#change line 1454, 1455 to spec_layer_params.scalingFactor.append(int(scaling_factor_h)),
#spec_layer_params.scalingFactor.append(int(scaling_factor_w)) in coremltools/models/neural_network/builder.py

parser = argparse.ArgumentParser()
parser.add_argument("--type", default=0, type=int, help="Coefficient of efficientdet. can be 0,1,2,3")
args = parser.parse_args()

model_name = "efficientdet_{}.{}"

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

device = torch.device('cpu')
params = Params(os.path.join("Yet-Another-EfficientDet-Pytorch", "projects", "coco.yml"))
model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=args.type, onnx_export=True,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales)).to(device)                   
model.backbone_net.model.set_swish(memory_efficient=False)
#Downloading and loading model weights
os.system("wget {}".format(eff_urls[args.type]))
model.load_state_dict(torch.load(eff_urls[args.type].split('/')[-1]))
model.eval()
dummy_input = torch.randn((1,3, model.input_sizes[args.type],model.input_sizes[args.type]), dtype=torch.float32).to(device)
model(dummy_input) #running one forward pass so that all dynamism in forward pass is made as static

# opset_version 11 is causing error
torch.onnx.export(model, dummy_input,
                  model_name.format(args.type, "onnx"),
                  verbose=True,
                  input_names=['input_image'],
                  opset_version=10)

onnx_model = onnx.load(model_name.format(args.type, "onnx"))
onnx.checker.check_model(onnx_model)

#image scaling parameters(not exact one but approximating)
image_scale = 1.0/(255.0*0.226)
red_bias = -0.485/0.226
green_bias = -0.456/0.226
blue_bias = -0.406/0.226

ml_model = onnx_coreml.convert(model_name.format(args.type, "onnx"), image_input_names=['input_image'], 
							preprocessing_args= {"image_scale": image_scale, "red_bias": red_bias, 
							"blue_bias": blue_bias, "green_bias": green_bias},
							minimum_ios_deployment_target='13')
#at the time of this code creation torch to onnx has an error because of which padding dimensions arent properly
#transferred to onnx. so chaning it manually in the mlmodel graph
for i,layer in enumerate(ml_model._spec.neuralNetwork.layers):
	if "pad" in layer.name.lower():
		print("Chaning pad of layer {}".format(layer.name))
		change_error_dimension(layer)


print("Total anchors", model.total_anchors)
change_effdet_output_names(ml_model._spec) #chaning output node names in mlmodel graph
add_squeeze_layer(ml_model._spec, "box_scores_pre", "box_scores", [model.total_anchors, 90], [1, model.total_anchors, 90])
add_squeeze_layer(ml_model._spec, "box_coordinates_pre", "box_coordinates", [model.total_anchors, 4], [1, model.total_anchors, 4])
add_slicestatic_layer(ml_model._spec, "box_coordinates", "box_loc_yx", [model.total_anchors, 2], [model.total_anchors, 4],
                      [0,0], [2147483647, 2], [True, True], [True, False])
add_slicestatic_layer(ml_model._spec, "box_coordinates", "box_loc_hw", [model.total_anchors, 2], [model.total_anchors, 4],
                      [0,2], [2147483647, 4], [True, False], [True, True])
add_constant_layer(ml_model._spec, "anchor_yx", [model.total_anchors, 2], model.anchor_data[:, [0,1]])
add_constant_layer(ml_model._spec, "anchor_hw", [model.total_anchors, 2], model.anchor_data[:, [2,3]])
# add_elementwise_layer(ml_model._spec, "scale_yx", ["box_loc_yx"], [[2034, 2]],  alpha=0.1, mode="multiply")
# add_elementwise_layer(ml_model._spec, "scale_hw", ["box_loc_hw"], [[2034, 2]],  alpha=0.2, mode="multiply")
add_unary(ml_model._spec, "hw_exp", "box_loc_hw", [model.total_anchors, 2], mode="exp")
add_elementwise_layer(ml_model._spec, "final_hw_rev", ["hw_exp", "anchor_hw"], [[model.total_anchors, 2], [model.total_anchors, 2]], mode="multiply")
add_reverse_layer(ml_model._spec, "final_wh", "final_hw_rev", [model.total_anchors, 2], [False, True])
add_elementwise_layer(ml_model._spec, "final_yx_pre", ["box_loc_yx", "anchor_hw"], [[model.total_anchors, 2], [model.total_anchors, 2]], mode="multiply")
add_elementwise_layer(ml_model._spec, "final_yx_rev", ["final_yx_pre", "anchor_yx"], [[model.total_anchors, 2], [model.total_anchors, 2]], mode="add")
add_reverse_layer(ml_model._spec, "final_xy", "final_yx_rev", [model.total_anchors, 2], [False, True])
add_concat_layer(ml_model._spec, "box_locations", ["final_xy", "final_wh"], [[model.total_anchors, 2], [model.total_anchors, 2]], [model.total_anchors, 4], 1)
ml_model._spec.description.output[0].name = "box_locations"
ml_model._spec.description.output[1].name = "box_scores"

# #adding nms and creating a pipeline model
nms_spec = coremltools.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3
for i in range(2):
    detection_output = ml_model._spec.description.output[i].SerializeToString()
    nms_spec.description.input.add() 
    nms_spec.description.input[i].ParseFromString(detection_output)
    nms_spec.description.output.add() 
    nms_spec.description.output[i].ParseFromString(detection_output)
    
nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

output_sizes = [90, 4] 
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType 
    ma_type.shapeRange.sizeRanges.add() 
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0 
    ma_type.shapeRange.sizeRanges[0].upperBound = -1 
    ma_type.shapeRange.sizeRanges.add() 
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i] 
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i] 
    del ma_type.shape[:]
    
nms = nms_spec.nonMaximumSuppression 
nms.confidenceInputFeatureName = "box_scores" 
nms.coordinatesInputFeatureName = "box_locations" 
nms.confidenceOutputFeatureName = "confidence" 
nms.coordinatesOutputFeatureName = "coordinates" 
nms.iouThresholdInputFeatureName = "iouThreshold" 
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

default_iou_threshold = 0.6
default_confidence_threshold = 0.5
nms.pickTop.perClass = True
nms.iouThreshold = default_iou_threshold 
nms.confidenceThreshold = default_confidence_threshold
labels = np.loadtxt("../coco_labels.txt", dtype=str, delimiter="\n") 
nms.stringClassLabels.vector.extend(labels)
nms_model = coremltools.models.MLModel(nms_spec) 

#creating the pipeline model comprising efficientdet and NMS
input_features = [("input_image", datatypes.Array(3,model.input_sizes[args.type],model.input_sizes[args.type])), ("iouThreshold", datatypes.Double()),
("confidenceThreshold", datatypes.Double())] #cannot directly pass imageType as input type here. 
output_features = [ "confidence", "coordinates"]
pipeline = Pipeline(input_features, output_features)

pipeline.add_model(ml_model._spec)
pipeline.add_model(nms_model._spec)
pipeline.spec.description.input[0].ParseFromString(ml_model._spec.description.input[0].SerializeToString())
pipeline.spec.description.input[1].type.isOptional = True
pipeline.spec.description.input[2].type.isOptional = True
pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

final_model = coremltools.models.MLModel(pipeline.spec) 
final_model.save(model_name.format(args.type, "mlmodel"))
















