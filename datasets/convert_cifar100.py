from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
import numpy as np
import pickle
from six.moves import xrange
from random import shuffle

LABELS_FILENAME = 'labels.txt'

def _get_output_filename(dataset_dir, split_name):
    return '%s/cifar100%s.tfrecord' % (dataset_dir, split_name)
  
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
  
def image_to_tfexample(image_data, image_format, class_id, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
                                                                'image/encoded': bytes_feature(image_data),
                                                                'image/format ': bytes_feature(image_format),
                                                                'image/class/label': int64_feature(class_id),
                                                                'image/height': int64_feature(height),
                                                                'image/width': int64_feature(width),}))
  
def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))
      
### main
labels = np.zeros([100], dtype=int)
#labels[1]  = 0;labels[32] = 1;labels[67] = 2;labels[73] = 3;labels[91] = 4
#labels[12] = 5;labels[17] = 6;labels[37] = 7;labels[68] = 8;labels[76] = 9
#labels[34]= 10;labels[63]= 11;labels[64]= 12;labels[66]= 13;labels[75]= 14
#labels[47]= 15;labels[52]= 16;labels[56]= 17;labels[59]= 18;labels[96]= 19
def run(dataset_dir):
    dataset_type = 'png'
    
    training_filename = _get_output_filename(dataset_dir, '_train')
    test_filename = _get_output_filename(dataset_dir, '_test')
    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(training_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    with tf.device('/cpu:0'):
        labeled = [0]*100
        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            with tf.Session('',config=tf.ConfigProto(log_device_placement=True)) as sess:
                dataset_len = 0
                with open('/home/dmsl/Documents/data/cifar100/train', 'rb') as fo:
                    img_queue = pickle.load(fo, encoding='bytes')
                s = list(range(img_queue[b'data'].shape[0]))
                shuffle(s)
                for n in s:
                    label = img_queue[b'fine_labels'][n]
                    
                    if labeled[label]<250:
                        labeled[label] += 1
                    else:
                        img_queue[b'fine_labels'][n] = 100
                        
                for n in xrange(img_queue[b'data'].shape[0]):
                    image = img_queue[b'data'][n]
                    image = np.transpose(image.reshape(3,32,32),(1,2,0))
                    label = img_queue[b'fine_labels'][n]
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(label), 32, 32)
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                     % (n+dataset_len , dataset_len+img_queue[b'data'].shape[0]))
                dataset_len += img_queue[b'data'].shape[0]
                    
        with tf.python_io.TFRecordWriter(test_filename) as tfrecord_writer:
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            with tf.Session('',config=tf.ConfigProto(log_device_placement=True)) as sess:
                dataset_len = 0
                with open('/home/dmsl/Documents/data/cifar100/test', 'rb') as fo:
                    img_queue = pickle.load(fo, encoding='bytes')
                
                for n in xrange(img_queue[b'data'].shape[0]):
                    image = img_queue[b'data'][n]
                    image = np.transpose(image.reshape(3,32,32),(1,2,0))
                    label = img_queue[b'fine_labels'][n]
#                    clabel = img_queue[b'coarse_labels'][n]
                    
#                    if (clabel == 1) | (clabel == 9) | (clabel == 12) | (clabel == 17):
#                        label = labels[label]
                    image_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})
                    example = image_to_tfexample(image_string, str.encode(dataset_type), int(label), 32, 32)
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write('\r>> Reading dataset images %d/%d' 
                                     % (n+dataset_len , dataset_len+img_queue[b'data'].shape[0]))
                dataset_len += img_queue[b'data'].shape[0]
                        
        print('\nFinished converting the dataset! %d'%dataset_len)

