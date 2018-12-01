from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
slim = tf.contrib.slim

from nets.Distillation import RAS

def ResNext_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.batch_norm],decay=0.9,zero_debias_moving_mean=True, scale=True, activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.convolution2d], activation_fn = None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            biases_initializer=None,
                            weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
            return arg_sc

def bottlenecklayer(x, depth, stride=1, num_split = 8, is_training= False, val = False, name=None):
    with tf.variable_scope(name):
        convs = []
        for i in range(num_split):
            with tf.variable_scope('node%d'%i):
                conv = slim.convolution2d(x, 64, kernel_size=[1, 1], scope='pintwise_conv', trainable=is_training, reuse=val)
                conv = slim.batch_norm(conv, scope='batch0', is_training=is_training, reuse = val)
                conv = slim.convolution2d(conv, 64, stride = stride, kernel_size=[3, 3], scope='group_conv', trainable=is_training, reuse=val)
                conv = slim.batch_norm(conv, scope='batch1', is_training=is_training, reuse = val)
                convs.append(conv)
        conv = tf.concat(convs,3)
        conv = slim.convolution2d(conv, depth, kernel_size=[1, 1], scope='transition', trainable=is_training, reuse=val)
        conv = slim.batch_norm(conv, activation_fn= None, scope='transition_batch', is_training=is_training, reuse = val)
        if stride == 2:
            x = slim.avg_pool2d(x,[2,2], scope = 'pool')
            x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [depth//4, depth//4]]) # [?, height, width, channel]
        
        return tf.nn.relu(x+conv)
        
   
def ResNext(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='ECCV'):
    end_points = {}
#    large = sio.loadmat('/home/dmsl/nas/backup1/personal_lsh/model/vgg13.mat')
    with tf.variable_scope('ECCV_small'):
        x = slim.convolution2d(image, 64, kernel_size=[3, 3], scope='conv0', trainable=is_training, reuse=val)
        std0 = slim.batch_norm(x, scope='batch0', is_training=is_training, reuse = val)
        
        std1 = bottlenecklayer(std0,  64, num_split = 2, stride=1, is_training= is_training, val = val, name='bottleneck0')
        std2 = bottlenecklayer(std1, 128, num_split = 2, stride=2, is_training= is_training, val = val, name='bottleneck1')
        std3 = bottlenecklayer(std2, 256, num_split = 2, stride=2, is_training= is_training, val = val, name='bottleneck2')
        
        fc = slim.avg_pool2d(std3,[8,8], scope = 'GAP')
        fc = slim.flatten(fc)
        logits = slim.fully_connected(fc, 100, activation_fn=None, trainable=is_training, scope = 'full3', reuse = val)
        
    if is_training:
        with tf.variable_scope('ECCV'):
            x = slim.convolution2d(image, 64, kernel_size=[3, 3], scope='conv0', trainable=False, reuse=val)
            teach0 = slim.batch_norm(x, scope='batch0', is_training=False, reuse = val)
            
            teach1 = bottlenecklayer(teach0,  64, stride=1, is_training= False, val = val, name='bottleneck0')
            teach2 = bottlenecklayer(teach1, 128, stride=2, is_training= False, val = val, name='bottleneck1')
            teach3 = bottlenecklayer(teach2, 256, stride=2, is_training= False, val = val, name='bottleneck2')
            
            fc = slim.avg_pool2d(teach3,[8,8], scope = 'GAP')
            fc = slim.flatten(fc)
            logits2 = slim.fully_connected(fc, 100, activation_fn=None, trainable=False, scope = 'full3', reuse = val)
            end_points['Logits2'] = logits2
            
            with tf.variable_scope('Distillation'):
                    end_points['Dist'] = RAS([std0,   std1,   std2,     std3],
                                             [teach0, teach1, teach2, teach3], num_DFV = 1)
    

    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
ResNext.default_image_size = 32
