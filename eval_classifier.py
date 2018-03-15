import tensorflow as tf
slim = tf.contrib.slim
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.python.training.saver import Saver
from tensorflow.python.training       import supervisor
from tensorflow import Session
from tensorflow import ConfigProto

import time
import numpy as np
import scipy.io as sio
import cv2
import glob, os
train_dir   =  '/home/dmsl/Documents/tf/svd/init/vgg13_finetune'
#train_dir   =  '/home/dmsl/Documents/tf/vdsr12'
dataset_dir = '/home/dmsl/Documents/data/tf/cifar100_per10'
dataset_name = 'cifar100'
preprocessing_name = 'cifar10'
model_name   = 'vgg16_vmat'
batch_size = 128
tf.logging.set_verbosity(tf.logging.INFO)
        
with tf.Graph().as_default():
    ## Load Dataset
    dataset = dataset_factory.get_dataset(dataset_name, 'train', dataset_dir)
    with tf.device('/device:CPU:0'):
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                  shuffle=True,
                                                                  num_readers = 1,
                                                                  common_queue_capacity=dataset.num_samples,
                                                                  common_queue_min=0)
        images, labels = provider.get(['image', 'label'])
    
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, False)
    images = image_preprocessing_fn(images)

#    batch_images, batch_labels = tf.train.batch([images, labels],
#                                            batch_size = batch_size,
#                                            num_threads = 1,
#                                            capacity = dataset.num_samples)
    batch_images, batch_labels = tf.train.shuffle_batch([images, labels],
                                                    batch_size = batch_size,
                                                    num_threads = 1,
                                                    capacity = dataset.num_samples,
                                                    min_after_dequeue = 0)
    batch_queue = slim.prefetch_queue.prefetch_queue([batch_images, batch_labels], capacity=dataset.num_samples)
    
    img, lb = batch_queue.dequeue()
#    img = tf.placeholder(tf.float32, shape=(None, 32,32,3))
    network_fn = nets_factory.get_network_fn(model_name)
    end_points = network_fn(img, is_training=False)
    output = end_points['Logits']
    
#    task1 = tf.to_int32(tf.argmax(end_points['Logits'], 1))
#    training_accuracy1 = slim.metrics.accuracy(task1, tf.to_int32(lb))
    
    def _get_init_fn(checkpoint_path, ignore_missing_vars=False):
        return slim.assign_from_checkpoint_fn(checkpoint_path,
                                              slim.get_variables_to_restore(),
                                              ignore_missing_vars = ignore_missing_vars)    
    variables_to_restore = slim.get_variables_to_restore()
    checkpoint_path = latest_checkpoint(train_dir)
    saver = Saver(variables_to_restore)
    config = ConfigProto()
    config.gpu_options.allow_growth=True
    sess = Session(config=config)
    sv = supervisor.Supervisor(logdir=checkpoint_path,
                               init_fn=_get_init_fn(checkpoint_path, ignore_missing_vars=True),
                               summary_op=None,
                               summary_writer=None,
                               global_step=None,
                               saver=None)
    correct = 0
    predict = 0
    with sv.managed_session(master='', start_standard_services=False, config=config) as sess:
#        saver.restore(sess, checkpoint_path)
        optim_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        
        a = sess.run(optim_vars[-4:])
        
        layer = {}
        name = ['conv1w','conv1b',
                'conv2w','conv2b',
                'conv3w','conv3b',
                'conv4w','conv4b',
                'conv5w','conv5b',
                'conv6w','conv6b',
                'conv7w','conv7b',
                'conv8w','conv8b',
                'conv9w','conv9b',
                'conv10w','conv10b',
                'fc1w','fc1b',
                'fc2w','fc2b',
                'fc3w','fc3b']
        
#        print (optim_vars)
#        names = []
#        for i in range(0,len(optim_vars)):
#            p = sess.run(optim_vars[i])
#            names.append(p)
#            if len(list(p.shape)) ==2:
#                p = p.reshape([1,1,p.shape[0],p.shape[1]])
#            if (len(list(p.shape)) ==1)&(name[i][:4]=='conv'):
#                p = p.reshape([1,1,1,p.shape[0]])
#            layer[name[i]] = p
##                
#        t = time.time()
#        predict = np.array([0,0], dtype = float)
        sv.start_queue_runners(sess)
        label = []
        for i in range(500):
            label += list(sess.run([lb]))
#        l = 0
#        for i in range(dataset.num_samples//batch_size):
#        out = []
#        imgs_paths = glob.glob(os.path.join('/home/dmsl/Documents/data/IMAX', '*.tif'))
#        for i in range(len(imgs_paths)):
#            image = cv2.imread(imgs_paths[i]).astype(np.float32)
#            conv0 = sess.run(end_points['f0'])
#            conv1 = sess.run(end_points['f1'])
#            conv2 = sess.run(end_points['f2'])
#            conv3 = sess.run(end_points['f3'])
#            out.append(sess.run(output, feed_dict={img:[image]}))
            
#            predict += task
#            correct += np.sum(np.where(p1 == l1, 1,0))
#        end_point = sess.run(end_points)
#        print (time.time()-t)
#        
#    accuracy = correct/(dataset.num_samples//batch_size*batch_size)
#    print (accuracy)
    
    sess.close()
sio.savemat('/home/dmsl/nas/backup1/personal_lsh/model/vgg13_img.mat',layer)
