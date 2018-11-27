# Mixed precision training of Alexnet in Tensorflow

This project is still under construction and subject to change.

## Overview
This is an implementation of Alexnet using Tensorflow. The goal is to obtain weights that could be used in a forward pass through the model using 16 bit floating point inputs and activations. The model is trained on the ILSVRC12 dataset.

The weights are stored in 32 bit floating point and are cast into 16 bit during the forward pass. The final output result is cast to float32 to calculate the loss. The gradients for the weights are then calculated using this loss, and the float32 weights are updated using these gradients. 

## Acknowledgements
The general methodology is based on the NVIDIA Deep Learning SDK Documentation on training with mixed precision. (https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#training_tensorflow)

I used and modified parts of Kuba Karczewski's code for managing the input datasets and outputting the trained weights. (https://github.com/jakubkarczewski/AlexNet/blob/master/alexnet.py)

A .binaryproto format of mean image was obtained using the script from the caffe repository for the ILSVRC12 dataset. https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh and converted to a npy array using Coderx7's code: https://gist.github.com/Coderx7/26eebeefaa3fb28f654d2951980b80ba.
