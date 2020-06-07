import urllib
import tfcoreml

def get_input_name(name):
    return name[:-2].split('/')[-1]

def get_output_name(name):
    return name[:-2].split('/')[-1]

imagenet_labels = urllib.request.urlopen("https://bitbucket.org/goutham98/images/downloads/imagenetLabels.txt")                               
imagenet_labels = [i.decode("utf-8").strip()[:-1].split("'")[1] for i in imagenet_labels]

# this function changes the input and output name of layers by finding the given names
def change_names(nn, old_input=None, new_input=None, old_output=None, new_output=None):
    for i in range(len(nn.layers)):
        if old_input:
            if len(nn.layers[i].input) > 0:
                if nn.layers[i].input[0] == old_input:
                    nn.layers[i].input[0] = new_input
        if old_output:
            if len(nn.layers[i].output) > 0:
                if nn.layers[i].output[0] == old_output:
                    nn.layers[i].output[0] = new_output