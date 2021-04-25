# cnn-architectures

Implementation of influential convolutional neural network architectures using the Keras functional API.

## Architectures

### AlexNet
[Paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | [Implementation](https://github.com/s-themis/cnn-architectures/blob/main/architectures/alexnet.py)\
*Year published:* 2012\
*Depth:* 8\
*Parameters:* 62.4M\
*Reported training time:* 5-6 days on two GTX 580 3GB GPUs\
*Key architectural points*:
* Kernels of decreasing size 11x11, 5x5, 3x3
* ReLU activations
* Local response normalization
* Overlapping max pooling for downsampling
* Dropout on fully-connected layers

### VGGNet (Configuration D - VGG16)
[Paper](https://arxiv.org/pdf/1409.1556.pdf) | [Implementation](https://github.com/s-themis/cnn-architectures/blob/main/architectures/vggnet.py)\
*Year published:* 2014\
*Depth:* 16\
*Parameters:* 138.4M\
*Reported training time:* 2-3 weeks on four Titan Black GPUs\
*Key architectural points*:
* Homogeneous convolutional blocks with decreasing feature map size and increasing number of filters
* Homogeneous kernels of size 3x3
* ReLU activations
* No local response normalization
* Non-overlapping max pooling for downsampling on each block
* Dropout on fully-connected layers

### ResNet (ResNet50 Variant)
[Paper](https://arxiv.org/pdf/1512.03385.pdf) | [Implementation](https://github.com/s-themis/cnn-architectures/blob/main/architectures/resnet.py)\
*Year published:* 2015\
*Depth:* 50\
*Parameters:* 25.6M\
*Reported training time:* -\
*Key architectural points*:
* Initial kernel size 7x7 followed by overlapping max-pooling for downsampling
* Homogenous stacks of residual blocks with decreasing feature map size and increasing number of filters
* Kernels of size 1x1, 3x3, 1x1 on each residual block to bottleneck number of filters on 3x3 layers
* ReLU activations
* Batch normalization before activations
* Downsampling with overlapping convolutions
* Global average pooling before single fully-connected layer
* No dropout
