import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow import ConfigProto
slim = tf.contrib.slim

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import scipy.io as sio
train_dir   =  '/home/dmsl/Documents/tf/svd/VGG/VGG'
dataset_dir = '/home/dmsl/Documents/data/tf/cifar100'

dataset_name = 'cifar100'
model_name   = 'VGG'
preprocessing_name = 'cifar100'

Optimizer = 'sgd' # 'adam' or 'sgd'
Learning_rate =1e-2

batch_size = 128
val_batch_size = 200
init_epoch = 0
num_epoch = 200+init_epoch
weight_decay = 1e-4

checkpoint_path = None
#checkpoint_path   =  '/home/dmsl/Documents/tf/svd/mobile/mobile'
ignore_missing_vars = True
### main
#%%    
tf.logging.set_verbosity(tf.logging.INFO)
def _get_init_fn(checkpoint_path, ignore_missing_vars):
    if checkpoint_path is None:
        return None
    variables_to_restore = slim.get_variables_to_restore()[1:]
    for v in variables_to_restore:
        print (v)
    
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        
    return slim.assign_from_checkpoint_fn(checkpoint_path,
                                          variables_to_restore,
                                          ignore_missing_vars = ignore_missing_vars)

def GET_dataset(dataset_name, dataset_dir, batch_size, preprocessing_name, split):
    if split == 'train':
        sff = True
        threads = 4
        is_training = True
    else:
        sff = False
        threads = 1
        is_training = False
    with tf.variable_scope('dataset_%s'%split):
        dataset = dataset_factory.get_dataset(dataset_name, split, dataset_dir)
        with tf.device('/device:CPU:0'):
            if split == 'train':
                global_step = slim.create_global_step()
                p = tf.floor_div(tf.cast(global_step, tf.float32), tf.cast(int(dataset.num_samples / float(batch_size)), tf.float32))
            else:
                global_step = None
                p = None
            provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                      shuffle=sff,
                                                                      num_readers = threads,
                                                                      common_queue_capacity=dataset.num_samples,
                                                                      common_queue_min=0)
        images, labels = provider.get(['image', 'label'])
        
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training)
        images = image_preprocessing_fn(images)
        if split == 'train':
            batch_images, batch_labels = tf.train.shuffle_batch([images, labels],
                                                    batch_size = batch_size,
                                                    num_threads = threads,
                                                    capacity = dataset.num_samples,
                                                    min_after_dequeue = 0)
            with tf.variable_scope('1-hot_encoding'):
                batch_labels = slim.one_hot_encoding(batch_labels, dataset.num_classes,on_value=1.0)
                
            batch_queue = slim.prefetch_queue.prefetch_queue([batch_images, batch_labels], capacity=40*batch_size)
            
            image, label = batch_queue.dequeue()
            
        else:
            batch_images, batch_labels = tf.train.batch([images, labels],
                                                         batch_size = batch_size,
                                                         num_threads = threads,
                                                         capacity = dataset.num_samples)
        
            with tf.variable_scope('1-hot_encoding'):
                batch_labels = slim.one_hot_encoding(batch_labels, dataset.num_classes,on_value=1.0)
            batch_queue = slim.prefetch_queue.prefetch_queue([batch_images, batch_labels], capacity=8*batch_size)
            
            image, label = batch_queue.dequeue()
    return p, global_step, dataset, image, label

def sigmoid(x,k):
    return 1/(1+tf.exp(-(x-k)))
            
def MODEL(model_name, weight_decay, image, label, lr, epoch, is_training):
    network_fn = nets_factory.get_network_fn(model_name, weight_decay = weight_decay)
    end_points = network_fn(image, is_training=is_training, lr = lr, val=not(is_training))
    losses = []    
    if is_training:
        def scale_grad(x, scale):
            return scale*x + tf.stop_gradient((1-scale)*x)
        
        with tf.variable_scope('Student_loss'): 
            loss = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
            accuracy = slim.metrics.accuracy(tf.to_int32(tf.argmax(end_points['Logits'], 1)),
                                     tf.to_int32(tf.argmax(label, 1)))
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            losses.append(loss+tf.add_n(tf.losses.get_regularization_losses()))
            
        with tf.variable_scope('Dist_loss'):
            dist_loss = end_points['Dist']
            tf.summary.scalar('dist_loss', dist_loss)
            losses.append(dist_loss)
            
    else:
        losses = tf.losses.softmax_cross_entropy(label,end_points['Logits'])
        accuracy = slim.metrics.accuracy(tf.to_int32(tf.argmax(end_points['Logits'], 1)),
                                         tf.to_int32(tf.argmax(label, 1)))    
    return losses, accuracy
#%%    
with tf.Graph().as_default() as graph:
    ## Load Dataset
    epoch, global_step, dataset, image, label = GET_dataset(dataset_name, dataset_dir,
                                                        batch_size, preprocessing_name, 'train')
    _, _, val_dataset, val_image, val_label = GET_dataset(dataset_name, dataset_dir,
                                                          val_batch_size, preprocessing_name, 'test')
    with tf.device('/device:CPU:0'):
        decay_steps = dataset.num_samples // batch_size
        max_number_of_steps = int(dataset.num_samples/batch_size*(num_epoch))
        
    total_loss, train_accuracy = MODEL(model_name, weight_decay, image, label, Learning_rate, epoch, True)
    
    #%% Compute Loss & Gradient
    def distillation_learning_rate(Learning_rate, epoch, init_epoch):
        Learning_rate = tf.case([
                                 (tf.less(epoch,100+init_epoch), lambda : Learning_rate),
                                 (tf.less(epoch,150+init_epoch), lambda : Learning_rate*1e-1),
                                 ],
                                 default =                       lambda : 0.0)
        tf.summary.scalar('learning_rate', Learning_rate)
        return Learning_rate
    
    variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    optimizer = tf.train.MomentumOptimizer(distillation_learning_rate(Learning_rate, epoch, init_epoch),
                                           0.9, use_nesterov=True)
    gradient0 = optimizer.compute_gradients(total_loss[0], var_list = variables)
    gradient1 = optimizer.compute_gradients(total_loss[1], var_list = variables)
    with tf.variable_scope('clip_grad1'):
        for i, grad in enumerate(gradient1):
            grad0 = grad[0]
            if grad[0] != None:
                norm = tf.sqrt(tf.reduce_sum(tf.square(gradient0[i][0])))*sigmoid(epoch,0)
                gradient0[i] = (gradient0[i][0] + tf.clip_by_norm(grad[0],norm),   gradient0[i][1])
    
    update_ops.append(optimizer.apply_gradients(gradient0, global_step=global_step))
    update_op = tf.group(*update_ops)
    train_op = control_flow_ops.with_dependencies([update_op], tf.add_n(total_loss), name='train_op')
    
    val_loss, val_accuracy = MODEL(model_name, weight_decay, val_image, val_label, Learning_rate, epoch, False)
    
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')    
    
    
    #%% for validation
    def ts_fn(session, *args, **kwargs):
        total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)
        if ( ts_fn.step % (ts_fn.decay_steps) == 0):
            accuracy = 0
            itr = val_dataset.num_samples//val_batch_size
            for i in range(itr):
                accuracy += session.run(ts_fn.val_accuracy)
            print ('Epoch %s Step %s - Loss: %.2f Accuracy: %.2f%%, Highest Accuracy : %.2f%%'
                   % (str((ts_fn.step-ts_fn.decay_steps*ts_fn.init_epoch)//ts_fn.decay_steps).rjust(3, '0'),
                      str(ts_fn.step-ts_fn.decay_steps*ts_fn.init_epoch).rjust(6, '0'),
                      total_loss, accuracy *100/itr, ts_fn.highest*100/itr))
            acc = tf.Summary(value=[tf.Summary.Value(tag="Accuracy", simple_value=accuracy*100/itr)])
            ts_fn.eval_writer.add_summary(acc, ts_fn.step-ts_fn.decay_steps*ts_fn.init_epoch)
            
#            if accuracy > ts_fn.highest:
            ts_fn.saver.save(session, "%s/best_model.ckpt"%train_dir)
            print ('save new parameters')
            ts_fn.highest = accuracy
            
        ts_fn.step += 1
        return [total_loss, should_stop] 
    
    ts_fn.saver = tf.train.Saver()
    ts_fn.eval_writer = tf.summary.FileWriter('%s/eval'%train_dir,graph,flush_secs=60)
    ts_fn.step = 0
    ts_fn.decay_steps = decay_steps
    ts_fn.init_epoch = init_epoch
    ts_fn.val_accuracy = val_accuracy
    ts_fn.highest = 0
    
    #%% training
    config = ConfigProto()
    config.gpu_options.allow_growth=True
    
    slim.learning.train(train_op, logdir = train_dir, global_step = global_step,
                        session_config = config,
                        init_fn=_get_init_fn(checkpoint_path, ignore_missing_vars),
                        summary_op = summary_op,
                        train_step_fn=ts_fn,
                        number_of_steps = max_number_of_steps,
                        log_every_n_steps =  40,                #'The frequency with which logs are print.'
                        save_summaries_secs = 120,                #'The frequency with which summaries are saved, in seconds.'
                        save_interval_secs = 0)               #'The frequency with which the model is saved, in seconds.'

