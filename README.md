# CoreML Conversion Hub

This repository contains the code for converting various deep learning models from Tensorflow and Pytorch to CoreML format. Converting models in standard frameworks like Tensorflow and Pytorch isn't always a straightforward process as the conversion libraries are still evolving and may have to change the code for different kinds of model types. In addition to the converted CoreML models, this repo also contains the code for converting them so that users can use them for converting some other models of similar types. Use the IOS app [AiBench](https://apps.apple.com/app/id1518857334) to benchmark all these provided models on real devices.

## Requirements

 - tensorflow == 2.2.0
 - coremltools == 3.4
 - onnx_coreml == 1.3
 - onnx == 1.7.0
 - torch == 1.5.1
## Conversion code
All the conversion codes are present under either Tensorflow or Pytorch directory with names mentioned in the below table. Run them to download the models in the original framework format from the source and convert them. Go through the code comments for understanding various model surgery performed for creating the model in correct format.
## CoreML Library
Classification models like mobilenet, efficientnet, resnet, inception are provided
### Classification
| Model Name | Download Link
|--|--|
| Mobilenet V1 | [mobilenetV1.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/mobilenetV1.mlmodel?raw=true) |
|Mobilenet V2 | [mobilenetV2.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/mobilenetV2.mlmodel?raw=true)|
|Resnet V2 |[resnetV2.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/resnetV2.mlmodel?raw=true) |
 |Inception V3| [inceptionV3.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/inceptionV3.mlmodel?raw=true)|

#### EfficientNet-Lite
| Model Name | Download Link |
|--|--|
| Eff-Lite-b0 | [efflite_b0.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efflite_b0.mlmodel?raw=true) |
| Eff-Lite-b1 | [efflite_b1.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efflite_b1.mlmodel?raw=true) |
| Eff-Lite-b2 | [efflite_b2.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efflite_b2.mlmodel?raw=true) |
| Eff-Lite-b3 | [efflite_b3.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efflite_b3.mlmodel?raw=true) |
| Eff-Lite-b4 | [efflite_b4.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efflite_b4.mlmodel?raw=true) |

#### EfficientNet-Edge-TPU
| Model Name | Download Link |
|--|--|
| Eff-Edge-Small | [eff_edgetpu_small.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/eff_edgetpu_small.mlmodel?raw=true) |
| Eff-Edge-Medium | [eff_edgetpu_medium.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/eff_edgetpu_medium.mlmodel?raw=true) |
| Eff-Edge-Large | [eff_edgetpu_large.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/eff_edgetpu_large.mlmodel?raw=true) |

#### MobileNet V3
| Model Name | Download Link |
|--|--|
| Mbnet-V3-Small | [mobilenetV3_small.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/mobilenetV3_small.mlmodel?raw=true) |
| Mbnet-V3-Large | [mobilenetV3_large.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/mobilenetV3_large.mlmodel?raw=true) |
| Mbnet-V3-Small-Minimal | [mobilenetV3_small_min.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/mobilenetV3_small_min.mlmodel?raw=true) |
| Mbnet-V3-Large-Minimal | [mobilenetV3_large_min.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/mobilenetV3_large_min.mlmodel?raw=true) |

### Detection
Mobilenet based SSD networks and EfficientDet models are provided.
| Model Name | Download Link |
|--|--|
|SSD-MbNet-V2| [MobileNetV2_SSDLite.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/MobileNetV2_SSDLite.mlmodel?raw=true) |
|SSD-MbNet-V3-Small| [ssd_mobilenet_v3_small.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/ssd_mobilenet_v3_small.mlmodel?raw=true) |
|SSD-MbNet-V3-Large| [ssd_mobilenet_v3_large.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/ssd_mobilenet_v3_large.mlmodel?raw=true) |

#### EfficienetDet
| Model Name | Download Link |
|--|--|
| EfficientDet-D0| [efficientdet_0.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efficientdet_0.mlmodel?raw=true) |
| EfficientDet-D1| [efficientdet_1.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efficientdet_1.mlmodel?raw=true) |
| EfficientDet-D2| [efficientdet_2.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efficientdet_2.mlmodel?raw=true) |
| EfficientDet-D3| [efficientdet_3.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/efficientdet_3.mlmodel?raw=true) |

### Face Analysis
| Model Name | Download Link |
|--|--|
| Blazeface-Mediapipe | [blazeface_pipeline.mlmodel](https://github.com/gouthamk1998/coremlmodels/blob/master/blazeface_pipeline.mlmodel?raw=true) |
| FaceMesh-Mediapipe|[facemesh.mlmodel](https://github.com/gouthamvgk/facemesh_coreml_tf/blob/master/coreml_models/facemesh.mlmodel?raw=true)|

## Credits

 - [https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
 - [https://github.com/tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
 - [https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
