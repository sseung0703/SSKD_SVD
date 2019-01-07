from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
import numpy as np

from nets.Distillation import RAS

slim = tf.contrib.slim

def VGG_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def VGG(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='vgg6'):
    end_points = {}
    with tf.variable_scope(scope, 'vgg6', [image]):
        with tf.variable_scope('block0'):
            teach0 = image
            for i in range(2):
                teach0 = slim.conv2d(teach0, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)        
            teach0 = slim.max_pool2d(teach0, [2, 2], 2, scope='pool')
            
        with tf.variable_scope('block1'):
            teach1 = teach0
            for i in range(2):
                teach1 = slim.conv2d(teach1, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)        
            teach1 = slim.max_pool2d(teach1, [2, 2], 2, scope='pool')
            
        with tf.variable_scope('block2'):
            teach2 = teach1
            for i in range(3):
                teach2 = slim.conv2d(teach2, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)        
            teach2 = slim.max_pool2d(teach2, [2, 2], 2, scope='pool')
            
        with tf.variable_scope('block3'):
            teach3 = teach2
            for i in range(3):
                teach3 = slim.conv2d(teach3, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)        
            teach3 = slim.max_pool2d(teach3, [2, 2], 2, scope='pool')
            
        fc = tf.contrib.layers.flatten(teach3)
        fc = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                   trainable=is_training, scope = 'full1', reuse = val)
        fc = slim.dropout(fc,is_training=is_training)
        fcs = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                   trainable=is_training, scope = 'full2', reuse = val)
        fc = slim.dropout(fcs,is_training=is_training)
        
        logits = slim.fully_connected(fc , 100, activation_fn=None,
                                      trainable=is_training, scope = 'full3', reuse = val)                
    end_points['Logits'] = logits
    return end_points
VGG.default_image_size = 32








