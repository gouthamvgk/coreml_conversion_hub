import tensorflow as tf
import coremltools
import os
import tfcoreml
import numpy as np
import urllib
import tarfile
from tensorflow.keras.layers import *
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from effedge_utils import * 
import sys
from argparse import ArgumentParser
sys.path.append("../")
from utils import *
print(tf.__version__)

parser = ArgumentParser()
parser.add_argument("--type", default="small", type=str, help="type of effEdgetpu to convert options are \
					small, medium , large")
args = parser.parse_args()

tar_url = eff_edgetpu_urls[args.type]
tar_file = tar_url.split("/")[-1]
folder = tar_file.split('.')[0]

os.system("wget {}".format(tar_url))
os.system("tar -xvf {}".format(tar_file))

cleaned_pb = "eff_edgetpu_{}.pb".format(args.type)
mlmodel_file = "eff_edgetpu_{}.mlmodel".format(args.type)

#Efficientnet edge tpu provides model in saved model format. directlu using it in coreml is not working. So first convert
#savedmodel to freezed graph and then use it with coreml

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], os.path.join(folder, "saved_model"))
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, # The session
            graph.as_graph_def(),
            ["Softmax"]
    )
with tf.compat.v1.gfile.GFile(cleaned_pb, "wb") as f:
    f.write(output_graph_def.SerializeToString())

#if we are converting from pb then we shouldn't strip the '/' from tensor names
output_graph_node = "Softmax"
input_graph_node = "images" #constant for all three models

eff_edgetpu_mlmodel = tfcoreml.convert(
cleaned_pb,
output_feature_names = [output_graph_node],
input_name_shape_dict = {input_graph_node: eff_input_size[args.type]},
image_input_names = [input_graph_node],
image_scale = 1/128,
red_bias = -0.9921875,
green_bias = -0.9921875,
blue_bias= -0.9921875,
predicted_probabilities_output = output_graph_node,
predicted_feature_name = "classLabels",
class_labels = ["background"] + imagenet_labels, #it has 1001 classes where 0 is background
minimum_ios_deployment_target='13'
)

spec = eff_edgetpu_mlmodel._spec #changing this spec will automatically reflect in mlmodel properties
old_input_name = spec.description.input[0].name
old_output_name = spec.description.output[0].name
spec.description.input[0].name = "ImageInput" #Old input name has reference at multiple places have to changle all
spec.neuralNetworkClassifier.preprocessing[0].featureName = "ImageInput" #have to change the preprocessor also

spec.description.output[0].name = "classProbs" #Old output name has reference at multiple places have to change all
spec.description.predictedProbabilitiesName = "classProbs"
spec.neuralNetworkClassifier.labelProbabilityLayerName = "classProbs"

change_names(spec.neuralNetworkClassifier, old_input_name, "ImageInput", old_output_name, "classProbs")

spec.description.input[0].shortDescription = "{}x{}x{} image input of model".format(*eff_input_size[args.type][1:])
spec.description.output[0].shortDescription = "Class to probability mapping dictionary"
spec.description.output[1].shortDescription = "Correct class label"
eff_edgetpu_mlmodel.short_description = "Efficient-edgetpu {} imagnet model".format(args.type)
eff_edgetpu_mlmodel.license = "Open source academic license"
eff_edgetpu_mlmodel.save(mlmodel_file)