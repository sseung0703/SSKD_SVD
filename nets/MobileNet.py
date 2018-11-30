from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
slim = tf.contrib.slim

from nets.Distillation import RAS

def MobileNet_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.batch_norm],decay=0.9,zero_debias_moving_mean=True, scale=True, activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.convolution2d, slim.fully_connected], activation_fn = None,
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            biases_initializer=None,
                            weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
            return arg_sc

def _depthwise_separable_conv(inputs, depth, stride=1,
                              is_training= False, val = False, name=None):
    with tf.variable_scope(name):
        depthwise_conv = slim.separable_convolution2d(inputs, stride=stride, scope='depthwise_conv', activation_fn = None,
                                                      num_outputs = None, depth_multiplier=1, kernel_size=[3, 3],
                                                      weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                      biases_initializer=None,
                                                      weights_regularizer=slim.l2_regularizer(1e-4),
                                                      trainable=is_training, reuse=val)
        bn = slim.batch_norm(depthwise_conv, scope='dw_batch_norm',
                             trainable=is_training, reuse=val)
        pointwise_conv = slim.convolution2d(bn, depth, scope='pointwise_conv',
                                            kernel_size=[1, 1], activation_fn = None,
                                            trainable=is_training, reuse=val)
        bn = slim.batch_norm(pointwise_conv, scope='pw_batch_norm',
                             trainable=is_training, reuse=val)
        return bn
        
   
def MobileNet(image, is_training=False, val = False, lr = None, prediction_fn=slim.softmax,scope='mobile'):
    end_points = {}
#    large = sio.loadmat('/home/dmsl/nas/backup1/personal_lsh/model/vgg13.mat')
    with tf.variable_scope('mobile_small'):
        net = slim.convolution2d(image, 32, scope='conv0',
                                 kernel_size=[3, 3], 
                                 trainable=is_training, reuse=val)
        std0 = slim.batch_norm(net, scope='bn0',
                              trainable=is_training, reuse=val)
        std0 = _depthwise_separable_conv(std0,  64, 1, is_training= is_training, val = val, name='mobile0')
        
        std1 = _depthwise_separable_conv(std0, 128, 2, is_training= is_training, val = val, name='mobile1')
        
        std2 = _depthwise_separable_conv(std1, 256, 2, is_training= is_training, val = val, name='mobile3')
        
        std3 = _depthwise_separable_conv(std2, 512, 2, is_training= is_training, val = val, name='mobile5')
        
        
        fc = slim.avg_pool2d(std3,[4,4], scope = 'GAP')
        fc = slim.flatten(fc)
        logits = slim.fully_connected(fc, 100, biases_initializer = tf.zeros_initializer(),
                                      trainable=is_training, scope = 'full3', reuse = val)
        
    if is_training:
        with tf.variable_scope('mobile'):
            net = slim.convolution2d(image, 32, scope='conv0',
                                     kernel_size=[3, 3], stride = 1,
                                     trainable=False, reuse=val)
            teach0 = slim.batch_norm(net, scope='bn0',
                                  trainable=False, reuse=val)
            teach0 = _depthwise_separable_conv(teach0,  64, 1, is_training= False, val = val, name='mobile0')
            
            teach1 = _depthwise_separable_conv(teach0, 128, 2, is_training= False, val = val, name='mobile1')
            teach1 = _depthwise_separable_conv(teach1, 128, 1, is_training= False, val = val, name='mobile2')
            
            teach2 = _depthwise_separable_conv(teach1, 256, 2, is_training= False, val = val, name='mobile3')
            teach2 = _depthwise_separable_conv(teach2, 256, 1, is_training= False, val = val, name='mobile4')
            
            teach3 = _depthwise_separable_conv(teach2, 512, 2, is_training= False, val = val, name='mobile5')
            teach3 = _depthwise_separable_conv(teach3, 512, 1, is_training= False, val = val, name='mobile6')
            
#            fc = slim.avg_pool2d(teach3,[4,4], scope = 'GAP')
#            fc = slim.flatten(fc)
#            logits = slim.fully_connected(fc, 100, biases_initializer = tf.zeros_initializer(), 
#                                          trainable=is_training, scope = 'full3', reuse = val)
        
            
            with tf.variable_scope('Distillation'):
                def gram(x,y,p=True,scope=scope):
                    with tf.variable_scope(scope):
                        sz0 = x.get_shape().as_list()
                        sz1 = y.get_shape().as_list()
                        if p:
                            x = slim.max_pool2d(x,[2,2])
                        x = tf.reshape(x,[sz0[0],-1,sz0[-1]])
                        
                        return tf.matmul(x,tf.reshape(y,[sz1[0],-1,sz1[-1]]),transpose_a=True)/(sz1[1]*sz1[2])
                   
                gs0 = gram(std0,std1,'gram_std0')
                gs1 = gram(std1,std2,'gram_std1')
                gs2 = gram(std2,std3,'gram_std2')
                
                gt0 = gram(teach0,teach1,'gram_teach0')
                gt1 = gram(teach1,teach2,'gram_teach1')
                gt2 = gram(teach2,teach3,'gram_teach2')
                
                end_points['Dist'] = tf.reduce_mean(tf.reduce_sum(tf.square(gs0-gt0),[1,2])
                                                  + tf.reduce_sum(tf.square(gs1-gt1),[1,2])
                                                  + tf.reduce_sum(tf.square(gs2-gt2),[1,2]))/(64*128+128*256+256*512)
                
#                end_points['Dist'] = RAS([std0,   std1,   std2,   std3,   std4],
#                                         [teach0, teach1, teach2, teach3, teach4], num_DFV = 1)
#    

    end_points['Logits'] = logits
    #end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return end_points
MobileNet.default_image_size = 32
