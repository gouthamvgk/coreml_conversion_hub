import tensorflow as tf
import os
import coremltools
from coremltools.models import datatypes
from coremltools.models.pipeline import *
import numpy as np
import tfcoreml
import sys
sys.path.append('../')
from utils import *
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes

tar_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz"
tar_file = tar_url.split("/")[-1]
folder = tar_file.split('.')[0]
cleaned_pb = "ssd_mobilenet_v3_small_cleaned.pb"
mlmodel_file = "ssd_mobilenet_v3_small.mlmodel"

os.system("wget {}".format(tar_url))
os.system("tar -xvf {}".format(tar_file))


with tf.compat.v1.gfile.GFile(os.path.join(folder, "frozen_inference_graph.pb"), "rb") as f: 
    graph_def = tf.compat.v1.GraphDef() 
    graph_def.ParseFromString(f.read()) 
#the pb graph has tensorflowlite detection process node which have to be removed as its not
#supported in coreml
output_def = tf.compat.v1.graph_util.extract_sub_graph(graph_def, ["Squeeze", "convert_scores"])
with tf.compat.v1.gfile.GFile(cleaned_pb, "wb") as f:
    f.write(output_def.SerializeToString())


ssd_mb_v3_small = tfcoreml.convert(
    cleaned_pb,
    output_feature_names = ["Squeeze", "convert_scores"],
    input_name_shape_dict = {"normalized_input_image_tensor": [1, 320, 320, 3]},
    image_input_names = ["normalized_input_image_tensor"],
    image_scale = 1/127.5,
    red_bias = -1,
    green_bias = -1,
    blue_bias= -1,
    minimum_ios_deployment_target='13'
)

#loading the tflite model and getting the prior anchor boxes from it
interpreter = tf.lite.Interpreter(model_path=os.path.join(folder, "model.tflite"))
interpreter.allocate_tensors()
anchors = interpreter.get_tensor(379) #the tensor index corresponding to anchor boxes prior
#use interpreter._get_ops_details() to know this details

#changing the input image node name
ssd_mb_v3_small._spec.neuralNetwork.preprocessing[0].featureName = "input_image"
ssd_mb_v3_small._spec.neuralNetwork.layers[0].input[0] = "input_image"
ssd_mb_v3_small._spec.description.input[0].name = "input_image"

#changing the node name of box location outputs
ssd_mb_v3_small._spec.neuralNetwork.layers[-3].name = "box_coordinates"
ssd_mb_v3_small._spec.neuralNetwork.layers[-3].output[0] = "box_coordinates"
ssd_mb_v3_small._spec.neuralNetwork.layers[-3].squeeze.axes.pop()
ssd_mb_v3_small._spec.neuralNetwork.layers[-3].squeeze.squeezeAll = True
ssd_mb_v3_small._spec.neuralNetwork.layers[-3].outputTensor[0].rank = 2
ssd_mb_v3_small._spec.neuralNetwork.layers[-3].outputTensor[0].dimValue.pop(0)


#Doing anchor box decoding
add_squeeze_layer(ssd_mb_v3_small._spec, "convert_scores", "box_scores", [2034, 91], [1, 2034, 91])
add_slicestatic_layer(ssd_mb_v3_small._spec, "box_coordinates", "box_loc_yx", [2034, 2], [2034, 4],
                      [0,0], [2147483647, 2], [True, True], [True, False])
add_slicestatic_layer(ssd_mb_v3_small._spec, "box_coordinates", "box_loc_hw", [2034, 2], [2034, 4],
                      [0,2], [2147483647, 4], [True, False], [True, True])
add_constant_layer(ssd_mb_v3_small._spec, "anchor_yx", [2034, 2], anchors[:, [0,1]])
add_constant_layer(ssd_mb_v3_small._spec, "anchor_hw", [2034, 2], anchors[:, [2,3]])
add_elementwise_layer(ssd_mb_v3_small._spec, "scale_yx", ["box_loc_yx"], [[2034, 2]],  alpha=0.1, mode="multiply")
add_elementwise_layer(ssd_mb_v3_small._spec, "scale_hw", ["box_loc_hw"], [[2034, 2]],  alpha=0.2, mode="multiply")
add_unary(ssd_mb_v3_small._spec, "hw_exp", "scale_hw", [2034, 2], mode="exp")
add_elementwise_layer(ssd_mb_v3_small._spec, "final_hw_rev", ["hw_exp", "anchor_hw"], [[2034, 2], [2034, 2]], mode="multiply")
add_reverse_layer(ssd_mb_v3_small._spec, "final_wh", "final_hw_rev", [2034, 2], [False, True])
add_elementwise_layer(ssd_mb_v3_small._spec, "final_yx_pre", ["scale_yx", "anchor_hw"], [[2034, 2], [2034, 2]], mode="multiply")
add_elementwise_layer(ssd_mb_v3_small._spec, "final_yx_rev", ["final_yx_pre", "anchor_yx"], [[2034, 2], [2034, 2]], mode="add")
add_reverse_layer(ssd_mb_v3_small._spec, "final_xy", "final_yx_rev", [2034, 2], [False, True])
add_concat_layer(ssd_mb_v3_small._spec, "box_locations", ["final_xy", "final_wh"], [[2034, 2], [2034, 2]], [2034, 4], 1)

#Chaning output description accordingly
ssd_mb_v3_small._spec.description.output[0].name = "box_scores"
ssd_mb_v3_small._spec.description.output[0].type.multiArrayType.shape.extend([2034, 91])
ssd_mb_v3_small._spec.description.output[0].type.multiArrayType.dataType = datatypes._FeatureTypes_pb2.ArrayFeatureType.DOUBLE
ssd_mb_v3_small._spec.description.output[1].name = "box_locations"
ssd_mb_v3_small._spec.description.output[1].type.multiArrayType.shape.extend([2034, 4])
ssd_mb_v3_small._spec.description.output[1].type.multiArrayType.dataType = datatypes._FeatureTypes_pb2.ArrayFeatureType.DOUBLE



#creating nms layer
nms_spec = coremltools.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3
for i in range(2):
    detection_output = ssd_mb_v3_small._spec.description.output[i].SerializeToString()
    nms_spec.description.input.add() 
    nms_spec.description.input[i].ParseFromString(detection_output)
    nms_spec.description.output.add() 
    nms_spec.description.output[i].ParseFromString(detection_output)
    
nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

output_sizes = [91, 4] 
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

#creating the pipeline model comprising mobilenet ssd and NMS
input_features = [("input_image", datatypes.Array(3,320, 320)), ("iouThreshold", datatypes.Double()),
("confidenceThreshold", datatypes.Double())] #cannot directly pass imageType as input type here. 
output_features = [ "confidence", "coordinates"]
pipeline = Pipeline(input_features, output_features)

pipeline.add_model(ssd_mb_v3_small._spec)
pipeline.add_model(nms_model._spec)
pipeline.spec.description.input[0].ParseFromString(ssd_mb_v3_small._spec.description.input[0].SerializeToString())
pipeline.spec.description.input[1].type.isOptional = True
pipeline.spec.description.input[2].type.isOptional = True
pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

final_model = coremltools.models.MLModel(pipeline.spec) 
final_model.save(mlmodel_file)



