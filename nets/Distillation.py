import tensorflow as tf
import numpy as np

def removenan_pair(X):
    with tf.variable_scope('remove_nan'):
        isfin_t = tf.logical_and(tf.is_finite(X[0]),tf.is_finite(X[1]))
        isfin_b = tf.logical_and(tf.is_finite(X[2]),tf.is_finite(X[3]))
        
        sz = X[0].get_shape().as_list()
        sh = list(range(len(sz)))[1:]
        
        isfin_batch = tf.reduce_min(tf.cast(isfin_t,tf.float32),sh,keep_dims=True)\
                     *tf.reduce_min(tf.cast(isfin_b,tf.float32),sh,keep_dims=True)
        mask = isfin_batch*sz[0]/(tf.reduce_sum(isfin_batch)+1e-3)
    
        X[0] = tf.where(isfin_t, X[0],tf.zeros_like(X[0]))
        X[1] = tf.where(isfin_t, X[1],tf.zeros_like(X[1]))
        X[2] = tf.where(isfin_b, X[2],tf.zeros_like(X[2]))
        X[3] = tf.where(isfin_b, X[3],tf.zeros_like(X[3]))
        
        X[0] = tf.stop_gradient((1 - mask) * X[0]) + mask * X[0]
        X[1] = tf.stop_gradient((1 - mask) * X[1]) + mask * X[1]
        X[2] = tf.stop_gradient((1 - mask) * X[2]) + mask * X[2]
        X[3] = tf.stop_gradient((1 - mask) * X[3]) + mask * X[3]
    
    return X

def SVD(X, n, name = None):
    with tf.variable_scope(name):
        sz = X.get_shape().as_list()
        num_decomposed = n + 1
        if len(sz)>2:
            x = tf.reshape(X,[sz[0],sz[1]*sz[2],sz[3]])
            num_decomposed = min(num_decomposed, sz[1]*sz[2], sz[3])
        else:
            x = tf.reshape(X,[sz[0],1,-1])
            num_decomposed = 1
        with tf.device('CPU'):
            s,u,v = tf.svd(x,full_matrices=False)

        s = tf.reshape(s,[sz[0],1,-1])
        
        s = tf.slice(s,[0,0,0],[-1,-1,num_decomposed])
        u = tf.slice(u,[0,0,0],[-1,-1,num_decomposed])
        v = tf.slice(v,[0,0,0],[-1,-1,num_decomposed])
        
        return s, u, v

def Align_rsv(x, y, y_s, k):
    x_sz = x.get_shape().as_list()
    
    y = tf.slice(y,[0,0,0],[-1,-1,k])
    y_s = tf.slice(y_s,[0,0,0],[-1,-1,k])
    
    x_ = tf.transpose(x,[0,2,1])
    x_temp = []
    r = tf.constant(np.array( list(range(x_sz[0])) ).reshape(-1,1,1),dtype=tf.int32)
    cosine = tf.matmul(x, y, transpose_a=True)
    index = tf.expand_dims(tf.cast(tf.argmax(tf.abs(cosine),1),tf.int32),-1)
    for i in range(k):
        idx = tf.slice(index,[0,i,0],[-1,1,-1])
        idx = tf.concat([r, idx],2)
        x_temp.append(tf.gather_nd(x_, idx ))
    
    x_temp = tf.concat(x_temp,1)
    x = tf.transpose(x_temp,[0,2,1])
    
    y_s = tf.stop_gradient(y_s/tf.sqrt(tf.reduce_sum(tf.square(y_s), [2], keep_dims=True)))
    x *= y_s; y*=y_s
    
    cosine = tf.expand_dims(tf.matrix_diag_part(tf.matmul(x,y,transpose_a=True)),1)
    sign = tf.where(tf.greater_equal(cosine,0.0), tf.ones_like(cosine),
                                                 -tf.ones_like(cosine))
    y *= sign
    
    return x, y
    
def Radial_Basis_Function(student, teacher, ts,n):
    loss = []
    for l in range(len(student)-1):
        with tf.variable_scope('RBF_node%d'%l):
            with tf.variable_scope('weighted_V'):
                svt, svb = student[l:l+2]
                tvt, tvb = teacher[l:l+2]
                
                t_sz = svt.get_shape().as_list()
                b_sz = svb.get_shape().as_list()
                tb_sz = tvb.get_shape().as_list()
                
                n = min(n, b_sz[2], tb_sz[2])
                tst, tsb = ts[l:l+2]
                
                svt, tvt = Align_rsv(svt,tvt, tst, n)
                svb, tvb = Align_rsv(svb,tvb, tsb, n)
                svt,tvt,svb,tvb = removenan_pair([svt,tvt,svb,tvb])
                        
            with tf.variable_scope('RBF'):
                svt = tf.reshape(svt,[t_sz[0],-1,1,n])
                svb = tf.reshape(svb,[b_sz[0],1,-1,n])
                
                tvt = tf.reshape(tvt,[t_sz[0],-1,1,n])
                tvb = tf.reshape(tvb,[b_sz[0],1,-1,n])
                
                s_rbf = tf.exp(-tf.square(svt-svb)/8)
                t_rbf = tf.exp(-tf.square(tvt-tvb)/8)
                rbf_loss = tf.nn.l2_loss((s_rbf-tf.stop_gradient(t_rbf)))*np.sqrt(n)
            
            loss.append(rbf_loss)
    
    return tf.add_n(loss)

def RAS(student_fm, teacher_fm, num_DFV=1):
    with tf.variable_scope('Distillation'):
        with tf.variable_scope('SVD'):
            student_rsv = []
            teacher_rsv = []
            teacher_sv = []
            for i in range(len(student_fm)):
                _,_,sv = SVD(student_fm[i], num_DFV, name='SVD_s%d'%i)
                ts,_,tv = SVD(teacher_fm[i], num_DFV, name='SVD_t%d'%i)
                student_rsv.append(sv)
                teacher_rsv.append(tv)
                teacher_sv.append(ts)
            
        with tf.variable_scope('RBF'):
            return Radial_Basis_Function(student_rsv, teacher_rsv, teacher_sv, num_DFV)
            
            
def crop_removenan(x,scale = True):
    isfin = tf.is_finite(x)
    sh = list(range(len(x.get_shape().as_list())))[1:]
    isfin_batch = tf.reduce_min(tf.cast(isfin,tf.float32),sh,keep_dims=True)
    
    x = tf.where(isfin, x,tf.zeros_like(x))
    x *= isfin_batch
    
    return x

def mmul(X):
    x = X[0]
    for i in range(1,len(X)):
        x = tf.matmul(x,X[i])
    return x
def msym(X):
    return (X+tf.matrix_transpose(X))/2

@tf.RegisterGradient('Svd')
def gradient_svd(op, ds, dU, dV):
    s, U, V = op.outputs
    u_sz = dU.get_shape().as_list()
    s_sz = ds.get_shape().as_list()
    v_sz = dV.get_shape().as_list()
    
    num_vec = 4
    k = min(num_vec + 1, s_sz[-1])
    
    s = tf.slice(s,[0,  0],[-1,   k])
    U = tf.slice(U,[0,0,0],[-1,-1,k])
    V = tf.slice(V,[0,0,0],[-1,-1,k])
    
    dU = tf.slice(dU,[0,0,0],[-1,-1,k])
    dV = tf.slice(dV,[0,0,0],[-1,-1,k])    
         
    if u_sz[1]<v_sz[1]:
        S = tf.matrix_diag(s)
        s_1 = tf.matrix_diag(1/s)
        s_2 = tf.square(s)
        
        k = crop_removenan(1.0/(tf.reshape(s_2,[s_sz[0],-1,1])-tf.reshape(s_2,[s_sz[0],1,-1])))
        K = tf.where(tf.eye(s_sz[-1],batch_shape=[s_sz[0]])==1.0, tf.zeros_like(k), k)
        
        D = tf.matmul(dV, s_1)
        grad = tf.matmul(U,D,transpose_b=True)\
             - tf.matmul(tf.matmul(U,tf.matrix_diag(tf.matrix_diag_part(tf.matmul(D,V,transpose_a=True)))), V,transpose_b=True)\
             - tf.matmul(2*tf.matmul(U, msym(K*tf.matmul(tf.matrix_transpose(tf.matmul(V,S)), D))), tf.matmul(V,S),transpose_b=True)
    else:
        S = tf.matrix_diag(s)
        s_2 = tf.square(s)
        
        k = crop_removenan(1.0/(tf.reshape(s_2,[s_sz[0],-1,1])-tf.reshape(s_2,[s_sz[0],1,-1])))
        KT = tf.matrix_transpose(tf.where(tf.eye(s_sz[-1],batch_shape=[s_sz[0]])==1.0, tf.zeros_like(k), k))
        
        grad = tf.matmul(2*mmul([U, S, msym(KT*tf.matmul(V,dV,transpose_a=True))]), V,transpose_b=True)
         
    return [crop_removenan(grad)]
