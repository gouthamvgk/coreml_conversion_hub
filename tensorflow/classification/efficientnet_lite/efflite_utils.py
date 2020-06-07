import tfcoreml

def get_output_tensor_index(operations):
	for i, k in enumerate(operations):
		if "softmax" in k.name.lower():
			print("Softmax output founded at index {}".format(i))
			print(k)
			return i

def get_input_tensor_index(operations):
	for i,k in enumerate(operations):
		if "truediv" in k.name.lower():
			print("Truediv input founded at index {}".format(i))
			print(k)
			return i

eff_lite_urls = {
	"b0": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz",
	"b1": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz",
	"b2": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite2.tar.gz",
	"b3": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite3.tar.gz",
	"b4": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz"
}

eff_input_size = {
	"b0": [1, 224, 224, 3],
	"b1": [1, 240, 240, 3],
	"b2": [1, 260, 260, 3],
	"b3": [1, 280, 280, 3],
	"b4": [1, 300, 300, 3],
}