import functools

import tensorflow as tf

from nets import VGG
from nets import VGG_teacher
from nets import MobileNet
from nets import ResNext

slim = tf.contrib.slim

networks_map   = {
                'VGG':VGG.VGG,
                'VGG_teacher':VGG_teacher.VGG,
                'MobileNet':MobileNet.MobileNet,
                'ResNext':ResNext.ResNext,
                 }

arg_scopes_map = {
                  'VGG':VGG.VGG_arg_scope,
                  'VGG_teacher':VGG_teacher.VGG_arg_scope,
                  'MobileNet':MobileNet.MobileNet_arg_scope,
                  'ResNext':ResNext.ResNext_arg_scope,
                 }



def get_network_fn(name, weight_decay=5e-4):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, is_training, lr = None, val = False):
        with slim.arg_scope(arg_scope):
            return func(images, is_training=is_training, lr = lr, val = val)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn

