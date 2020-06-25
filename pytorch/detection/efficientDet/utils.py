import os
from coremltools.models import datatypes
from coremltools.models.pipeline import *
import coremltools

eff_urls = {
    0 : "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth",
    1 : "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth",
    2 : "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth",
    3 : "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth"
}

def change_error_dimension(layer_spec):
    old_pad = layer_spec.constantPad.padAmounts  
    dim = len(old_pad)                     
    new_pad = [0, 0, 0, 0, old_pad[3], old_pad[7], old_pad[2], old_pad[6]]  
    for i in range(dim):
        layer_spec.constantPad.padAmounts.pop()                             
    layer_spec.constantPad.padAmounts.extend(new_pad)   

def change_effdet_output_names(model_spec):
    output1 = model_spec.description.output[0].name
    model_spec.description.output[0].type.multiArrayType.dataType = datatypes._FeatureTypes_pb2.ArrayFeatureType.DOUBLE
    output2 = model_spec.description.output[1].name
    model_spec.description.output[1].type.multiArrayType.dataType = datatypes._FeatureTypes_pb2.ArrayFeatureType.DOUBLE
    new_name1 = "box_coordinates_pre"
    new_name2 = "box_scores_pre"
    model_spec.description.output[0].name = new_name1
    model_spec.description.output[1].name = new_name2
    for lay in model_spec.neuralNetwork.layers:
        if lay.output[0] == output1:
            lay.output[0] = new_name1
        elif lay.output[0] == output2:
            lay.output[0] = new_name2


def add_squeeze_layer(spec, input_name, output_name, output_dims, input_dims):
    spec.neuralNetwork.layers.add()
    spec.neuralNetwork.layers[-1].squeeze.MergeFromString(b'')
    spec.neuralNetwork.layers[-1].name = output_name
    spec.neuralNetwork.layers[-1].input.append(input_name)
    spec.neuralNetwork.layers[-1].inputTensor.add()
    spec.neuralNetwork.layers[-1].inputTensor[0].rank = len(input_dims)
    spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend(input_dims)
    spec.neuralNetwork.layers[-1].outputTensor.add()
    spec.neuralNetwork.layers[-1].outputTensor[0].rank = len(output_dims)
    spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend(output_dims)
    spec.neuralNetwork.layers[-1].squeeze.squeezeAll = True
    spec.neuralNetwork.layers[-1].output.append(output_name)
    

def add_slicestatic_layer(spec, input_name, output_name, output_dims, input_dims, begin_id, end_id,
                         begin_mask, end_mask):
    spec.neuralNetwork.layers.add()
    spec.neuralNetwork.layers[-1].sliceStatic.MergeFromString(b'')
    spec.neuralNetwork.layers[-1].name = output_name
    spec.neuralNetwork.layers[-1].input.append(input_name)
    spec.neuralNetwork.layers[-1].inputTensor.add()
    spec.neuralNetwork.layers[-1].inputTensor[0].rank = len(input_dims)
    spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend(input_dims)
    spec.neuralNetwork.layers[-1].outputTensor.add()
    spec.neuralNetwork.layers[-1].outputTensor[0].rank = len(output_dims)
    spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend(output_dims)
    spec.neuralNetwork.layers[-1].sliceStatic.strides.extend([1,1])
    spec.neuralNetwork.layers[-1].sliceStatic.beginIds.extend(begin_id)
    spec.neuralNetwork.layers[-1].sliceStatic.endIds.extend(end_id)
    spec.neuralNetwork.layers[-1].sliceStatic.beginMasks.extend(begin_mask)
    spec.neuralNetwork.layers[-1].sliceStatic.endMasks.extend(end_mask)
    spec.neuralNetwork.layers[-1].output.append(output_name)
    

def add_constant_layer(spec, output_name, output_dims, constant_data):
    spec.neuralNetwork.layers.add()
    spec.neuralNetwork.layers[-1].loadConstantND.MergeFromString(b'')
    spec.neuralNetwork.layers[-1].loadConstantND.shape.extend(output_dims)
    spec.neuralNetwork.layers[-1].loadConstantND.data.floatValue.extend(map(float, constant_data.flatten()))
    spec.neuralNetwork.layers[-1].name = output_name
    spec.neuralNetwork.layers[-1].output.append(output_name)
    spec.neuralNetwork.layers[-1].outputTensor.add()
    spec.neuralNetwork.layers[-1].outputTensor[0].rank = 2
    spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend(output_dims)
    
def add_elementwise_layer(spec, output_name, input_names, inputs_dims, alpha=None, mode="multiply"):
    if len(input_names) == 1 and (not alpha):
        raise ValueError("Should provide alpha value when only one input is provided")
    if len(input_names) == 2 and alpha:
        raise ValueError("Alpha should be provided only with one input")
    spec.neuralNetwork.layers.add()
    if mode == "multiply":
        spec.neuralNetwork.layers[-1].multiply.MergeFromString(b'')
    elif mode == "add":
        spec.neuralNetwork.layers[-1].add.MergeFromString(b'')
    spec.neuralNetwork.layers[-1].input.extend(input_names)
    spec.neuralNetwork.layers[-1].output.append(output_name)
    spec.neuralNetwork.layers[-1].name = output_name
    for k, i in enumerate(inputs_dims):
        spec.neuralNetwork.layers[-1].inputTensor.add()
        spec.neuralNetwork.layers[-1].inputTensor[k].rank = len(i)
        spec.neuralNetwork.layers[-1].inputTensor[k].dimValue.extend(i)
    spec.neuralNetwork.layers[-1].outputTensor.add()
    spec.neuralNetwork.layers[-1].outputTensor[0].rank = len(inputs_dims[0])
    spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend(inputs_dims[0])
    if len(inputs_dims) == 1: spec.neuralNetwork.layers[-1].multiply.alpha = alpha
        
def add_unary(spec, output_name, input_name, input_dims, mode="exp"):
    spec.neuralNetwork.layers.add()
    spec.neuralNetwork.layers[-1].unary.MergeFromString(b'')
    spec.neuralNetwork.layers[-1].unary.shift = 0
    spec.neuralNetwork.layers[-1].unary.scale = 1
    spec.neuralNetwork.layers[-1].unary.epsilon = 1e-6
    spec.neuralNetwork.layers[-1].input.append(input_name)
    spec.neuralNetwork.layers[-1].output.append(output_name)
    spec.neuralNetwork.layers[-1].name = output_name
    spec.neuralNetwork.layers[-1].inputTensor.add()
    spec.neuralNetwork.layers[-1].inputTensor[0].rank = len(input_dims)
    spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend(input_dims)
    if mode == "exp":
        spec.neuralNetwork.layers[-1].unary.type = coremltools.proto.NeuralNetwork_pb2.UnaryFunctionLayerParams.Operation.Value('EXP')
    elif mode == "log":
        spec.neuralNetwork.layers[-1].unary.type = coremltools.proto.NeuralNetwork_pb2.UnaryFunctionLayerParams.Operation.Value('LOG')
    elif mode == "abs":
        spec.neuralNetwork.layers[-1].unary.type = coremltools.proto.NeuralNetwork_pb2.UnaryFunctionLayerParams.Operation.Value('ABS')
    else:
        raise ValueError("Mode not understood")
    spec.neuralNetwork.layers[-1].outputTensor.add()
    spec.neuralNetwork.layers[-1].outputTensor[0].rank = len(input_dims)
    spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend(input_dims)
    
def add_concat_layer(spec, output_name, input_names, input_dims, output_dims, axis):
    spec.neuralNetwork.layers.add()
    spec.neuralNetwork.layers[-1].concatND.MergeFromString(b'')
    spec.neuralNetwork.layers[-1].input.extend(input_names)
    spec.neuralNetwork.layers[-1].output.append(output_name)
    spec.neuralNetwork.layers[-1].name = output_name
    for k, i in enumerate(input_dims):
        spec.neuralNetwork.layers[-1].inputTensor.add()
        spec.neuralNetwork.layers[-1].inputTensor[k].rank = len(i)
        spec.neuralNetwork.layers[-1].inputTensor[k].dimValue.extend(i)
    spec.neuralNetwork.layers[-1].outputTensor.add()
    spec.neuralNetwork.layers[-1].outputTensor[0].rank = len(output_dims)
    spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend(output_dims)
    spec.neuralNetwork.layers[-1].concatND.axis = axis
    
def add_reverse_layer(spec, output_name, input_name, input_dims, axis):
    spec.neuralNetwork.layers.add()
    spec.neuralNetwork.layers[-1].reverse.MergeFromString(b'')
    spec.neuralNetwork.layers[-1].reverse.reverseDim.extend(axis)
    spec.neuralNetwork.layers[-1].input.append(input_name)
    spec.neuralNetwork.layers[-1].output.append(output_name)
    spec.neuralNetwork.layers[-1].name = output_name
    spec.neuralNetwork.layers[-1].inputTensor.add()
    spec.neuralNetwork.layers[-1].inputTensor[0].rank = len(input_dims)
    spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend(input_dims)
    spec.neuralNetwork.layers[-1].outputTensor.add()
    spec.neuralNetwork.layers[-1].outputTensor[0].rank = len(input_dims)
    spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend(input_dims)