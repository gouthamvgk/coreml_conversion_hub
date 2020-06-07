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
import sys
sys.path.append("../")
from utils import *
print(tf.__version__)

resnet_50 = tf.keras.applications.ResNet50V2(
    include_top=True, weights='imagenet', input_tensor = Input(shape=(224, 224, 3), batch_size=1))
resnet_50.save("resnet_50.h5")

# CoreMl tools will automatically consider the model as NeuralNetwork classifier if you provide the 
# class_labels argument with the class names of classification model
resnet_50_mlmodel = tfcoreml.convert(
"./resnet_50.h5",
output_feature_names = [get_output_name(resnet_50.outputs[0].name)],
input_name_shape_dict = {get_input_name(resnet_50.inputs[0].name): list(resnet_50.inputs[0].shape)},
image_input_names = [get_input_name(resnet_50.inputs[0].name)],
image_scale = 1/127.5,
red_bias = -1,
green_bias = -1,
blue_bias=-1,
predicted_probabilities_output = get_output_name(resnet_50.outputs[0].name),
predicted_feature_name = "classLabels",
class_labels = imagenet_labels,
minimum_ios_deployment_target='13'
)

#Chaning the input name and one of the output name for convenience. Also adding model descriptions
spec = resnet_50_mlmodel._spec #changing this spec will automatically reflect in mlmodel properties
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
resnet_50_mlmodel.short_description = "Resnet V2 imagnet model"
resnet_50_mlmodel.license = "Open source academic license"
resnet_50_mlmodel.save("resnetV2.mlmodel")