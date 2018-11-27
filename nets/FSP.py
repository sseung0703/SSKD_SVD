import tensorflow as tf
slim = tf.contrib.slim

def Grammian(top, bot):
    t_sz = top.get_shape().as_list()
    b_sz = bot.get_shape().as_list()

    if t_sz[1] > b_sz[1]:
        top = slim.max_pool2d(top, [2, 2], 2)
                        
    top = tf.reshape(top,[t_sz[0], -1, t_sz[-1]])
    bot = tf.reshape(bot,[b_sz[0], -1, b_sz[-1]])

    Gram = tf.matmul(top, bot, transpose_a = True)/(b_sz[1]*b_sz[2])
    return Gram, t_sz[-1]*b_sz[-1]

def FSP(students, teachers):
    #students : list of student feature map
    #teachers : list of teacher feature map
    # reuturn : Distillation loss
    N = 0                    
    Dist_loss = []
    for i in range(len(students)-1):
        gs0, _ = Grammian(students[i], students[i+1])
        gt0, n = Grammian(teachers[i], teachers[i+1])
 
        Dist_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(gt0)-gs0),[1,2])/2 ))
        N += n

    return tf.add_n(Dist_loss)/N 