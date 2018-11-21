#!/usr/bin/env python3
# This piece of script is obtained from
# https://github.com/jakubkarczewski/AlexNet/blob/master/alexnet.py
#
#
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Implementation to train Alexnet in 16-bit floating point precision
# TO DO:
#   1. Run code


# What to set for these variables, in the context of float16?
#   - stddev for random initialization of variables, used paper value of 0.01
#   - bias initialization: used default zero, different from paper
#   - LRN: depth_radius: 2/2/5, bias: 1/1/2, alpha: 2e-5/1e-5/1e-4
#           (ours/alexnet_train/paper)

#Other considerations:
#   1. Read caffe code to learn more how it trains the network
#   2. How to use weight decay together with MomentumOptimizer?


import tensorflow as tf
import os
#import psycopg2
#import urllib.request
import argparse
import sys
import cv2
#import matplotlib.pyplot as plt
import random
import numpy as np
from datetime import datetime
import time

import pickle

""
# The following custom getter function is obtained from:
# https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#training_tensorflow
def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

class my_float16_variable(object):
    def __init__(self, name, shape, stddev):
        self.var_float32 = tf.get_variable(
            name + "_float32",
            shape=shape,
            initializer=tf.truncated_normal_initializer( #?? What is this?
                        stddev=stddev,
                        dtype=tf.float32),
            collections=["float32_vars"]
        )

        self.var_float16 = tf.get_variable(
            name + "_float16",
            initializer=tf.zeros(shape, dtype=tf.float16),
            collections=["float16_vars"]
        )

    def get_float16(self):
        # cast from underlying fp32 to fp16 whenever get_float16 is called
        self.var_float16 = tf.cast(self.var_float32, tf.float16)
        return self.var_float16
    def get_float32(self):
        return self.var_float32


def variable_with_random_init(name, shape, stddev):
    var = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer( #?? What is this?
                        stddev=stddev,
                        dtype=tf.float16),
            dtype=tf.float16,
            trainable=True)
    # if wd is not None:
    #     weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    #     tf.add_to_collection('losses', weight_decay)
    return var

# Max pooling layer
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

# Drop-out layer
def dropout(x, keepPro, name = None):
    return tf.nn.dropout(x, keepPro, name=name)

# LRN layer
def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

# Fully connect layer
# Apply a clipping to the layer's bound.
# Due to the susequent application of ReLU, there is no need to clip a minimum, we clip it to -3000 just as is.
# def fcLayer_bound(x, inputD, outputD, reluFlag, name, bound):
#     """fully-connect"""
#     with tf.variable_scope(name) as scope:
#         w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
#         b = tf.get_variable("b", [outputD], dtype = "float")
#         out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
#         clipped_out = tf.clip_by_value(out, -3000, bound[name[:3]])
#         if reluFlag:
#             return tf.nn.relu(clipped_out)
#         else:
#             return clipped_out

# Fully connect layer with no bound
def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w_init = my_float16_variable("w", [inputD, outputD], 0.01)
        w = w_init.get_float16()
        # w = variable_with_random_init("w", [inputD, outputD], 0.01)
        ## w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b_init = my_float16_variable("b", [outputD], 0.01)
        b = b_init.get_float16()
        # b = tf.get_variable("b", [outputD], dtype = tf.float16)
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
    convolve = lambda a, b: tf.nn.conv2d(
        a, b,
        strides = [1, strideY, strideX, 1],
        padding = padding)
    with tf.variable_scope(name) as scope:
        w_init = my_float16_variable(
            "w",
            [kHeight, kWidth, channel/groups, featureNum],
            0.01)
        w = w_init.get_float16()
        #w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b_init = my_float16_variable("b", [featureNum], 0.01)
        b = b_init.get_float16()
        #b = tf.get_variable("b", shape = [featureNum], dtype = tf.float16)

        if groups == 1:
            conv = convolve(x, w)
        else:
            xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)#input and weights after split
            wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)
            output_groups = [convolve(t1, t2) for t1, t2 in zip(xNew, wNew)] #retriving the feature map separately
            conv = tf.concat(axis = 3, values = output_groups) #concatnating feature map
        # print mergeFeatureMap.shape
        bias = tf.reshape(tf.nn.bias_add(conv, b), tf.shape(conv))
        # Clip value
        # clipped_out = tf.clip_by_value(out, -3000, bound[name[:5]])
        return tf.nn.relu(bias, name = scope.name)


class AlexNet_train(object):
    """alexNet model"""
    def __init__(self, x, keepPro, classNum = 1000):
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

        self.conv4 = convLayer(self.conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        self.conv5 = convLayer(self.conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        self.pool5 = maxPoolLayer(self.conv5, 3, 3, 2, 2, "pool5", "VALID")

        self.fcIn = tf.reshape(self.pool5, [-1, 256 * 6 * 6])
        self.fc6 = fcLayer(self.fcIn, 256 * 6 * 6, 4096, True, "fc6")
        self.dropout1 = dropout(self.fc6, self.KEEPPRO)

        self.fc7 = fcLayer(self.dropout1, 4096, 4096, True, "fc7")
        self.dropout2 = dropout(self.fc7, self.KEEPPRO)

        self.fc8 = fcLayer(self.dropout2, 4096, self.CLASSNUM, False, "fc8")

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
        return self.fc8

class Dataset:
    ''' Class for handling Imagenet data '''

    def __init__(self, image_path, mean_file_name, do_train, use_auxdata = False, auxfile_path = None):
        # For mean_file_name, convert imagenet_mean.binaryproto to npy format
        # first using an external tool, such as
        # https://gist.github.com/Coderx7/26eebeefaa3fb28f654d2951980b80ba.
        self.use_auxdata = use_auxdata
        self.do_train = do_train
        if not use_auxdata:
            print("-- Creating image list ...")
            self.data = create_image_list(image_path)
            print("-- Shuffling data ...")
            np.random.shuffle(self.data)
            self.num_records = len(self.data)
            self.next_record = 0

            self.labels, self.inputs = zip(*self.data)

            category = np.unique(self.labels)
            self.num_labels = len(category)
            self.category2label = dict(zip(category, range(len(category))))
            self.label2category = {l: k for k, l in self.category2label.items()}

            # Convert the labels to numbers
            self.labels = [self.category2label[l] for l in self.labels]
            print("-- Finished loading dataset for train data")

        else:
            self.data = []
            print("-- Creating image list ...")
            with open(auxfile_path, 'r') as f:
                for line in f:
                    filename, category_num = line.split(' ')
                    category_num = int(category_num)
                    filename = os.path.join(image_path, filename)
                    self.data.append([category_num, filename])
            print("-- Shuffling data...")
            np.random.shuffle(self.data)
            self.num_records = len(self.data)
            self.next_record = 0
            self.labels, self.inputs = zip(*self.data)
            self.num_labels = len(np.unique(self.labels))
            print("-- Finished loading dataset for validation data")

        print("-- Got", self.num_records, "data entries and", self.num_labels, "different labels")

        input_mean = np.load(mean_file_name)
        self.mean = np.zeros((input_mean.shape[1], input_mean.shape[2], input_mean.shape[0]))
        for c in range(self.mean.shape[2]):
            self.mean[:,:,c] = input_mean[c,:,:]


    def __len__(self):
        return self.num_records

    def onehot(self, label):
        v = np.zeros(self.num_labels)
        print(type(label))
        v[label] = 1
        return v

    def records_remaining(self):
        return len(self) - self.next_record

    def has_next_record(self):
        return self.next_record < self.num_records

    def preprocess(self, img_file_name):
        img = cv2.imread(img_file_name)
        if len(img.shape) == 2:
            #duplicate colors in all channels
            new_pp = np.zeros(img.shape[0], img.shape[1], 3)
            new_pp[:,:,0] = img
            new_pp[:,:,1] = img
            new_pp[:,:,2] = img
            img = new_pp

        h = img.shape[0]
        w = img.shape[1]
        dest_size = 227
        if h != 256 or w != 256:
            # crop image - keep smaller dimension, and crop the middle part of
            # the longer dimension
            if h > w:
                cropped = img[(h//2-w//2):(h//2+w//2),:,:]
            elif w > h:
                cropped = img[:,(w//2-h//2):(w//2+h//2),:]
            else:
                cropped = img
        # resize image into 256x256
        cropped = cv2.resize(cropped, (256,256))
        # subtract mean from image
        pp = cropped - self.mean
        offset = 0

        if self.do_train:
            # randomly crop 227x227 square from the 256x256 square when training
            offset = np.random.random_integers(0, 256 - dest_size)
        else:
            # only crop the center 227x227 square when testing
            offset = (256 - dest_size) // 2
        pp = pp[offset:(offset+dest_size), offset:(offset+dest_size), :]
        pp = np.asarray(pp, dtype=np.float16)
        return pp

    def next_record_f(self):
        if not self.has_next_record():
            # start from the beginning, reshuffle data
            np.random.shuffle(self.data)
            self.next_record = 0
            self.labels, self.inputs = zip(*self.data)

            if not self.use_auxdata:
                category = np.unique(self.labels)
                self.num_labels = len(category)
                self.category2label = dict(zip(category, range(len(category))))
                self.label2category = {l: k for k, l in self.category2label.items()}

                # Convert the labels to numbers
                self.labels = [self.category2label[l] for l in self.labels]

        # return None
        label = self.onehot(self.labels[self.next_record])
        input = self.preprocess(self.inputs[self.next_record])
        self.next_record += 1
        return label, input

    def next_batch(self, batch_size):
        records = []
        for i in range(batch_size):
            record = self.next_record_f()
            if record is None:
                break
            records.append(record)
        labels, input = zip(*records)
        return labels, input


def create_image_list(image_path):
    image_filenames = []
    category_list = [c for c in sorted(os.listdir(image_path))
                     if c[0] != '.' and
                     os.path.isdir(os.path.join(image_path, c))]
    for category in category_list:
        if category:
            walk_path = os.path.join(image_path, category)
        else:
            walk_path = image_path
            category = os.path.split(image_path)[1]

        w = _walk(walk_path)
        while True:
            try:
                dirpath, dirnames, filenames = next(w)
            except StopIteration:
                break
            # Don't enter directories that begin with '.'
            for d in dirnames[:]:
                if d.startswith('.'):
                    dirnames.remove(d)
            dirnames.sort()
            # Ignore files that begin with '.'
            filenames = [f for f in filenames if not f.startswith('.')]
            filenames.sort()

            for f in filenames:
                image_filenames.append([category, os.path.join(dirpath, f)])

    return image_filenames


def _walk(top):
    ''' Improved os.walk '''
    names = os.listdir(top)
    dirs, nondirs = [], []
    for name in names:
        if os.path.isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    yield top, dirs, nondirs
    for name in dirs:
        path = os.path.join(top, name)
        for x in _walk(path):
            yield x

def check_finiteness(grads):
    for g in grads:
        if not np.all(np.isfinite(g)):
            return False
    return True

# def total_cost(logits, labels):
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#         logits=logits, labels=labels), name='cross_entropy')
#     tf.add_to_collection('losses', cross_entropy)
#     return tf.add_n(tf.get_collection('losses'), name='total_cost')
"""
print('Loading data')
# training = Dataset('/local/train', 'mean.npy', True, False)
testing = Dataset('/local/val_images', 'mean.npy', False, True, 'val.txt')
print('Data loaded.')
# train_label, train_input = training.next_record_f()
test_label, test_input = testing.next_record_f()
"""
def main(_):
    # here we train and validate the model
    """
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: alexnet.py [--training_epoch=x] '
              '[--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.training_epoch <= 0:
        print('Please specify a positive value for training iteration.')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)
    """
    use_pickle = True
    create_pickle = False 
    training_pickle_file = 'training.p'
    testing_pickle_file = 'testing.p'

    training_epoch = 1
    model_version = 0.01
    export_dir = "alexout"

    if not use_pickle:
        print('Loading data')
        training = Dataset('/local/train', 'mean.npy', True, False)
        testing = Dataset('/local/val_images', 'mean.npy', False, True, 'val.txt')
        if create_pickle:
            print('Creating pickle files for training and testing data...')
            with open(training_pickle_file, 'wb') as train_f:
                pickle.dump(training, train_f)
            with open(testing_pickle_file, 'wb') as test_f:
                pickle.dump(testing, test_f)
            print('Finished creating pickles.')
        print('Data loaded.')
    else:
        print('Loading data from pickle...')
        with open(training_pickle_file, 'rb') as train_f:
            training = pickle.load(train_f)
        with open(testing_pickle_file, 'rb') as test_f:
            testing = pickle.load(test_f)
        print("Finished loading data from pickle")

    batch_size = 128
    display_step = 20
    # training_acc_step = 1000 # think how to use it
    train_size = len(training)
    n_classes = training.num_labels
    image_size = 227
    img_channel = 3
    num_epochs = training_epoch
    initial_scale_factor = 128.

    # -- TRAINING SETUP -- #
    # -- Setting up placeholders
    x_flat = tf.placeholder(tf.float16, ##### Let input be float16
                            (None, image_size * image_size * img_channel))
    x_3d = tf.reshape(x_flat, shape=(tf.shape(x_flat)[0], image_size,
                                     image_size, img_channel))
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float16)
    scale_factor = tf.placeholder(tf.float32)

    model = AlexNet_train(x_3d, keep_prob, classNum=n_classes)
    # -- cast logits to float32 to calculate loss

    output_logits = tf.cast(model.output(), tf.float32)

    model_prediction = tf.nn.softmax(output_logits)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=y))
    #-- Loss scaling
    scaled_loss = tf.multiply(loss, tf.cast(scale_factor, tf.float32))
    # Cast loss to fp16 to ensure outputs of gradient calculation is fp16

    # -- Setting up optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')

    lr = tf.train.exponential_decay(0.01, global_step, 100000, 0.1,
                                    staircase=True)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=lr,
        momentum=0.9)

    # -- Step 1, get list of gradients
    float32_vars = tf.get_collection("float32_vars")

    float32_grads = tf.gradients(scaled_loss, float32_vars)

    # -- Step 2, check if there are NaN or inf variables in grads

    # -- Step 3, cast grads to fp32 and downscale
    float32_grads = [tf.divide(g, scale_factor) for g in float32_grads]

    grads_and_vars = zip(float32_grads, float32_vars)

    # -- Step 4, apply weight update
    apply_gradient_op = optimizer.apply_gradients(
        grads_and_vars,
        global_step=global_step
    )

    # optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(cost, global_step=global_step)

    accuracy, update_op = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                              predictions=tf.argmax(model_prediction,
                                                                    1))
    test_accuracy, test_update_op = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                                        predictions=tf.argmax(
                                                            model_prediction, 1))

    start_time = time.time()
    print("Start time is: " + str(start_time))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        step = 0
        steps_factor_unchanged = 0
        curr_scale_factor = initial_scale_factor

        while step < int(num_epochs * train_size) // batch_size:

            batch_ys, batch_xs = training.next_batch(batch_size)

            train_feed_dict = {
                x_3d: batch_xs,
                y: batch_ys,
                keep_prob: 0.5,
                scale_factor: curr_scale_factor
            }
            test_feed_dict = {
                x_3d: batch_xs,
                y: batch_ys,
                keep_prob: 1.,
                scale_factor: curr_scale_factor
            }

            float16_grads = sess.run(grads, feed_dict=train_feed_dict)
            # Check if there are invalid gradients here
            if check_finiteness(float16_grads):
                # sess.run(optimizer, feed_dict={x_3d: batch_xs, y: batch_ys, keep_prob: 0.5})
                sess.run(lr)
                if step % display_step == 0:
                    acc_up = sess.run([accuracy, update_op], feed_dict=test_feed_dict)
                    acc = sess.run(accuracy, feed_dict=test_feed_dict)
                    loss = sess.run(cost, feed_dict=test_feed_dict)
                    elapsed_time = time.time() - start_time
                    print(" Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                    ", Training Accuracy= " + "{}".format(acc) + " Elapsed time:" + str(elapsed_time) + \
                    "acc_up={}".format(acc_up))
                step += 1
                steps_factor_unchanged += 1
                if steps_factor_unchanged >= 500:
                    steps_factor_unchanged = 0
                    curr_scale_factor *= 2
            else:
                steps_factor_unchanged = 0
                curr_scale_factor /= 2

        print("Optimization Finished!")
        print("Training took" + str(time.time() - start_time))

        step_test = 1
        acc_list = []
        while step_test * batch_size < len(testing):
            testing_ys, testing_xs = testing.next_batch(batch_size)
            acc_up = sess.run([test_accuracy, test_update_op],
                              feed_dict={x_3d: testing_xs, y: testing_ys, keep_prob: 1.})
            acc = sess.run([test_accuracy],
                           feed_dict={x_3d: testing_xs, y: testing_ys, keep_prob: 1.})
            acc_list.append(acc)
            print("Testing Accuracy:", acc)
            step_test += 1

        print("Max accuracy is", max(acc_list))
        print("Min accuracy is", min(acc_list))

        # save model using SavedModelBuilder from TF
        # export_path_base = sys.argv[-1]
        export_path_base = export_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(x_flat)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(output_logits)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(),
                                  name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()

        print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
