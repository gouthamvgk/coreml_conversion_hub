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
from efflite_utils import * 
import sys
from argparse import ArgumentParser
sys.path.append("../")
from utils import *
print(tf.__version__)

parser = ArgumentParser()
parser.add_argument("--type", default="b0", type=str, help="type of efflite to convert options are \
					b0, b1, b2, b3, b4")
args = parser.parse_args()

tar_url = eff_lite_urls[args.type]
tar_file = tar_url.split("/")[-1]
folder = tar_file.split('.')[0]

os.system("wget {}".format(tar_url))
os.system("tar -xvf {}".format(tar_file))

initial_pb = "efflite_{}.pb".format(args.type)
cleaned_pb = "efflite_{}_cleaned.pb".format(args.type)
mlmodel_file = "efflite_{}.mlmodel".format(args.type)

#Efficient lite doesn't provide model in pb, h5 or saved format. It is in either tflite of ckpt.meta format of tf v1
#So let us convert the ckpt to pb format and then convert it to coreml
#Lot of unwanted initialized and saver variables are present inside the meta. have to remove them
#using strip_unused directly will remove some essential nodes so first using a different method to create
#a pb with data loader then cleaning that pb in next step
tf.compat.v1.disable_eager_execution() #have to disable ee for importing meta graph
graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph(os.path.join(folder, 'model.ckpt.meta'), clear_devices=True)
sess.run(tf.compat.v1.global_variables_initializer())
saver.restore(sess, os.path.join(folder, 'model.ckpt'))
input_graph_def = graph.as_graph_def()
out_index = get_output_tensor_index(sess.graph.get_operations())
output_node = sess.graph.get_operations()[out_index].outputs[0].name[:-2] #this varies for different version of efflite 
print("Output Node: ", output_node)
output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes 
            output_node.split(",")  
)
with tf.compat.v1.gfile.GFile(initial_pb, "wb") as f:
    f.write(output_graph_def.SerializeToString())

tf.compat.v1.reset_default_graph()
with tf.Graph().as_default() as graph: 
    tf.import_graph_def(output_graph_def, 
    input_map=None, 
    return_elements=None, 
    name="" 
)

operations = graph.get_operations()
input_index = get_input_tensor_index(operations)
input_graph_node = operations[input_index].outputs[0].name[:-2]
output_graph_node = operations[-1].outputs[0].name[:-2]
print(input_graph_node, output_graph_node)
reduced_graph = strip_unused_lib.strip_unused(input_graph_def = graph.as_graph_def(),
                                input_node_names=[input_graph_node],
                                output_node_names=[output_graph_node],
                                placeholder_type_enum=dtypes.float32.as_datatype_enum)
with tf.compat.v1.gfile.GFile(cleaned_pb, "wb") as f:
    f.write(reduced_graph.SerializeToString())

tf.compat.v1.enable_eager_execution()
#if we are converting from pb then we shouldn't strip the '/' from tensor names
eff_lite_mlmodel = tfcoreml.convert(
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
class_labels = imagenet_labels, 
minimum_ios_deployment_target='13'
)

spec = eff_lite_mlmodel._spec #changing this spec will automatically reflect in mlmodel properties
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
eff_lite_mlmodel.short_description = "Efficient-lite {} imagnet model".format(args.type)
eff_lite_mlmodel.license = "Open source academic license"
eff_lite_mlmodel.save(mlmodel_file)