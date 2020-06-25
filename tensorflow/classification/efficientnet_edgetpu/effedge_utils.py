import tfcoreml


eff_edgetpu_urls = {
	"small": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-edgetpu-S.tar.gz",
	"medium": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-edgetpu-M.tar.gz",
	"large": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-edgetpu-L.tar.gz"
}

eff_input_size = {
	"small": [1, 224, 224, 3],
	"medium": [1, 240, 240, 3],
	"large": [1, 300, 300, 3]
}