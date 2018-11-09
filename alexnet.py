##############################
# Implementation to train Alexnet in 16-bit floating point precision
# TO DO:
# 1. Find how to use weight decay with variables: read caffe code
#    https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# 2. Understand what truncated_normal_initializer does
# 3. What to set for these variables?
#    wd for each variable
#    stddev for random initialization of variables
#    LRN: keep intact (?)
# 4. Replace variable defs in conv and fc layers using this
# 5. Read caffe code to learn more how it trains the network
# 6. Consider the possibility of not training the network but quantizing the
# 7. trained activations

import tensorflow as tf
import os
import psycopg2
import urllib.request
import argparse
import sys
import cv2
#import matplotlib.pyplot as plt
import random
import numpy as np
from datetime import datetime
now = datetime.now()

def variable_with_weight_decay(name, shape, wd, randstddev):
    var = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer( #?? What is this?
                        stddev=stddev,
                        dtype=tf.float32)),
            dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')

# Max pooling layer
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

# Drop-out layer
def dropout(x, keepPro, name = None):
    return tf.nn.dropout(x, keepPro, name)

# LRN layer
def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

# Fully connect layer
# Apply a clipping to the layer's bound.
# Due to the susequent application of ReLU, there is no need to clip a minimum, we clip it to -3000 just as is.
def fcLayer(x, inputD, outputD, reluFlag, name, bound):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        clipped_out = tf.clip_by_value(out, -3000, bound[name[:3]])
        if reluFlag:
            return tf.nn.relu(clipped_out)
        else:
            return clipped_out

# Fully connect layer with no bound
def fcLayer_nb(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

# convlayer that is not clipped
# Filter has the shape [kHeight, kWidth, num_of_input_channels, featureNUM],
# where featureNUM is the output # of channels
def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1): #group=2 means the second part of AlexNet
    """convlutional"""
    channel = int(x.get_shape()[-1]) #get channel
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)#input and weights after split
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)] #retriving the feature map separately
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap) #concatnating feature map
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
        # Clip value
        # clipped_out = tf.clip_by_value(out, -3000, bound[name[:5]])
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)


class alexNet_train(object):
    """alexNet model"""
    def __init__(self, x, keepPro, classNum):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        #build CNN
        self.buildCNN()

    # Building the network
    def buildCNN(self):
        """build model"""
        self.conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        # def       LRN(x, R, alpha, beta, name = None, bias = 1.0):
        self.lrn1 = LRN(self.conv1, 2, 2e-05, 0.75, "norm1")
        # def          maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME")
        self.pool1 = maxPoolLayer(self.lrn1, 3, 3, 2, 2, "pool1", "VALID")

        self.conv2 = convLayer(self.pool1, 5, 5, 1, 1, 256, "conv2", groups = 2)

        # def convLayer(x, kHeight, kWidth, strideX, strideY,
        #          featureNum, name, padding = "SAME", groups = 1)

        self.lrn2 = LRN(self.conv2, 2, 2e-05, 0.75, "lrn2")
        self.pool2 = maxPoolLayer(self.lrn2, 3, 3, 2, 2, "pool2", "VALID")

        self.conv3 = convLayer(self.pool2, 3, 3, 1, 1, 384, "conv3")

        self.conv4 = convLayer(self.conv3, 3, 3, 1, 1, 384, "conv4" groups = 2)

        self.conv5 = convLayer(self.conv4, 3, 3, 1, 1, 256, "conv5" groups = 2)
        self.pool5 = maxPoolLayer(self.conv5, 3, 3, 2, 2, "pool5", "VALID")

        self.fcIn = tf.reshape(self.pool5, [-1, 256 * 6 * 6])
        self.fc6 = fcLayer(self.fcIn, 256 * 6 * 6, 4096, True, "fc6", self.bound)
        self.dropout1 = dropout(self.fc6, self.KEEPPRO)

        self.fc7 = fcLayer(self.dropout1, 4096, 4096, True, "fc7", self.bound)
        self.dropout2 = dropout(self.fc7, self.KEEPPRO)

        self.fc8 = fcLayer_nb(self.dropout2, 4096, self.CLASSNUM, False, "fc8")

        self.layers = {
            'conv1': self.conv1,
            'lrn1': self.lrn1,
            'pool1': self.pool1,
            'conv2': self.conv2,
            'lrn2': self.lrn2,
            'pool2': self.pool2,
            'conv3': self.conv3,
            'conv4': self.conv4,
            'conv5': self.conv5,
            'pool5': self.pool5,
            'fcIn': self.fcIn,
            'fc6': self.fc6,
            'dropout1': self.dropout1,
            'fc7': self.fc7,
            'dropout2': self.dropout2,
            'fc8': self.fc8
        }

    # A handy function to get the activation of certain layer
    def get_act(self, layer_name='fc8'):
        #print(self.layers[layer_name])
        return self.layers[layer_name]

    def output(self):
        return self.layers['fc8']

NUM_CLASSES = 1000

x = tf.placeholder(tf.float32, [1,227,227,3])
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
dropoutPro = tf.placeholder(tf.float32)

image = cv2.imread('/home/yiizy/val_images/ILSVRC2012_val_' + str(img_num).zfill(8) + '.JPEG')
test = cv2.resize(image.astype(float), (227, 227))
imgMean = np.array([104, 117, 124], np.float)
test -= imgMean
test_image = test.reshape((1, 227, 227, 3)) #reshape into tensor shape

model = alexNet(x, dropoutPro, NUM_CLASSES)
