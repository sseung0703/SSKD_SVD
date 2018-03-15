from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
import numpy as np

from nets.Distillation import RAS

slim = tf.contrib.slim

def vgg16_vmat_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc

def vgg16_vmat(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='vgg13_std'):
    end_points = {}
    large = sio.loadmat('/home/dmsl/nas/backup1/personal_lsh/model/vgg13.mat')
#    large = sio.loadmat('/home/dmsl/nas/backup1/personal_lsh/model/imgnet.mat')
    with tf.variable_scope(scope, 'vgg16_std', [image]):
        with tf.variable_scope('block0'):
            std0 = image
            for i in range(1):
                std0 = slim.conv2d(std0, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)        
            std0 = slim.max_pool2d(std0, [2, 2], 2, scope='pool')   
        with tf.variable_scope('block1'):
            std1 = std0
            for i in range(1):
                std1 = slim.conv2d(std1, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)        
            std1 = slim.max_pool2d(std1, [2, 2], 2, scope='pool')   
        with tf.variable_scope('block2'):
            std2 = std1
            for i in range(1):
                std2 = slim.conv2d(std2, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)      
            std2 = slim.max_pool2d(std2, [2, 2], 2, scope='pool')
            
        with tf.variable_scope('block3'):
            std3 = std2
            for i in range(1):
                std3 = slim.conv2d(std3, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                   scope='conv%d'%i, trainable=is_training, reuse=val)        
            std3 = slim.max_pool2d(std3, [2, 2], 2, scope='pool')
#                    
        fc = tf.contrib.layers.flatten(std3)
#        fc = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
##                                    weights_initializer = tf.constant_initializer(large['fc1w']),
##                                    biases_initializer  = tf.constant_initializer(large['fc1b']),
#                                   trainable=is_training, scope = 'full1', reuse = val)
#        fc = slim.dropout(fc,is_training=is_training)
        fcs = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
#                                   weights_initializer = tf.constant_initializer(large['fc2w']),
#                                   biases_initializer  = tf.constant_initializer(large['fc2b']),
                                   trainable=is_training, scope = 'full2', reuse = val)
        fc = slim.dropout(fcs,is_training=is_training)
        
        logits = slim.fully_connected(fc , 100, activation_fn=None,
                                      trainable=is_training, scope = 'full3', reuse = val)
         
        
        if is_training:
#        if True:
            with tf.variable_scope('teacher'):
                online_learning = False
                with tf.variable_scope('block0'):
                    teach0 = image
                    for i in range(2):
                        teach0 = slim.conv2d(teach0, 64, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['conv%dw'%(i+1)]),
                                           biases_initializer  = tf.constant_initializer(large['conv%db'%(i+1)]),
                                           scope='conv%d'%i, trainable=online_learning, reuse=val)        
                    teach0 = slim.max_pool2d(teach0, [2, 2], 2, scope='pool')
                    
                with tf.variable_scope('block1'):
                    teach1 = teach0
                    for i in range(2):
                        teach1 = slim.conv2d(teach1, 128, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['conv%dw'%(i+3)]),
                                           biases_initializer  = tf.constant_initializer(large['conv%db'%(i+3)]),
                                           scope='conv%d'%i, trainable=online_learning, reuse=val)        
                    teach1 = slim.max_pool2d(teach1, [2, 2], 2, scope='pool')
                    
                with tf.variable_scope('block2'):
                    teach2 = teach1
                    for i in range(3):
                        teach2 = slim.conv2d(teach2, 256, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['conv%dw'%(i+5)]),
                                           biases_initializer  = tf.constant_initializer(large['conv%db'%(i+5)]),
                                           scope='conv%d'%i, trainable=online_learning, reuse=val)        
                    teach2 = slim.max_pool2d(teach2, [2, 2], 2, scope='pool')
                    
                with tf.variable_scope('block3'):
                    teach3 = teach2
                    for i in range(3):
                        teach3 = slim.conv2d(teach3, 512, [3, 3], 1, padding = 'SAME', activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['conv%dw'%(i+8)]),
                                           biases_initializer  = tf.constant_initializer(large['conv%db'%(i+8)]),
                                           scope='conv%d'%i, trainable=online_learning, reuse=val)        
                    teach3 = slim.max_pool2d(teach3, [2, 2], 2, scope='pool')
                
                fc = tf.contrib.layers.flatten(teach3)
                fc = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                           weights_initializer = tf.constant_initializer(large['fc1w']),
                                           biases_initializer  = tf.constant_initializer(large['fc1b']),
                                           trainable=online_learning, scope = 'fullt1', reuse = val)
                fct = slim.fully_connected(fc, 1024, activation_fn=tf.nn.relu,
                                          weights_initializer = tf.constant_initializer(large['fc2w']),
                                          biases_initializer  = tf.constant_initializer(large['fc2b']),
                                          trainable=online_learning, scope = 'fullt2', reuse = val)
                logits2 = slim.fully_connected(fct, 100, activation_fn=None,
                                               trainable=online_learning, scope = 'fullt3', reuse = val)
                end_points['Logits2'] = logits2
                
                
                end_points['Dist'] = RAS([std0,   std1,   std2,   std3,   fcs],
                                         [teach0, teach1, teach2, teach3, fct], num_DFV = 4)
                
    end_points['Logits'] = logits
    return end_points
vgg16_vmat.default_image_size = 32








