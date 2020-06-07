import tensorflow as tf
import coremltools
import tfcoreml
import numpy as np
import urllib
import tarfile
import os
from tensorflow.keras.layers import *
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
import sys
sys.path.append("../")
from utils import *
print(tf.__version__)

#different v3 models can be found at https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
#mobilenetv3_large conversion
mobilenetv3_large_url = "https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz"
os.system("wget {}".format(mobilenetv3_large_url))
os.system("tar -xvf v3-large_224_1.0_float.tgz")

mbv3_large_pb = "./v3-large_224_1.0_float/v3-large_224_1.0_float.pb"
with tf.compat.v1.gfile.GFile(mbv3_large_pb, "rb") as f: 
    graph_def = tf.compat.v1.GraphDef() 
    graph_def.ParseFromString(f.read()) 

with tf.Graph().as_default() as graph: 
    tf.import_graph_def(graph_def, 
    input_map=None, 
    return_elements=None, 
    name="" 
)

#mobilenet_v3 has some unwanted identity nodes at beginning so removing them
operations = graph.get_operations()
input_graph_node = operations[2].outputs[0].name[:-2]
output_graph_node = operations[-1].outputs[0].name[:-2]
reduced_graph = strip_unused_lib.strip_unused(input_graph_def = graph.as_graph_def(),
                                input_node_names=[input_graph_node],
                                output_node_names=[output_graph_node],
                                placeholder_type_enum=dtypes.float32.as_datatype_enum)

with tf.compat.v1.gfile.GFile("mbv3_large_cleaned.pb", "wb") as f:
    f.write(reduced_graph.SerializeToString())

#if we are converting from pb then we shouldn't strip the '/' from tensor names
mobilenetv3_large_mlmodel = tfcoreml.convert(
"mbv3_large_cleaned.pb",
output_feature_names = [output_graph_node],
input_name_shape_dict = {input_graph_node: [1, 224, 224, 3]},
image_input_names = [input_graph_node],
image_scale = 1/127.5,
red_bias = -1,
green_bias = -1,
blue_bias=-1,
predicted_probabilities_output = output_graph_node,
predicted_feature_name = "classLabels",
class_labels = ["background"]+imagenet_labels, #mobilenetv3 in tf repo has 1001 class where first one is background
minimum_ios_deployment_target='13'
)


spec = mobilenetv3_large_mlmodel._spec #changing this spec will automatically reflect in mlmodel properties
old_input_name = spec.description.input[0].name
old_output_name = spec.description.output[0].name
spec.description.input[0].name = "ImageInput" #Old input name has reference at multiple places have to changle all
spec.neuralNetworkClassifier.preprocessing[0].featureName = "ImageInput" #have to change the preprocessor also

spec.description.output[0].name = "classProbs" #Old output name has reference at multiple places have to change all
spec.description.predictedProbabilitiesName = "classProbs"
spec.neuralNetworkClassifier.labelProbabilityLayerName = "classProbs"

change_names(spec.neuralNetworkClassifier, old_input_name, "ImageInput", old_output_name, "classProbs")

spec.description.input[0].shortDescription = "224x224x3 image input of model"
spec.description.output[0].shortDescription = "Class to probability mapping dictionary"
spec.description.output[1].shortDescription = "Correct class label"
mobilenetv3_large_mlmodel.short_description = "Mobilenet V3_large imagnet model"
mobilenetv3_large_mlmodel.license = "Open source academic license"
mobilenetv3_large_mlmodel.save("mobilenetV3_large.mlmodel")