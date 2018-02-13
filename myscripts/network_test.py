import tensorflow as tf
import numpy as np

N = 1000
I = 6
H = 10
O = 1

X = np.random.uniform(-5,5,(N,I))
Yx = np.sum(np.sin(X[:,::2]) + np.cos(X[:,1::2]), axis = 1)[...,None]

Z = np.random.uniform(-5,5,(N,H))
Yz = np.sum(Z[:,:-1:2] - Z[:,1::2], axis = 1)[...,None]

x = tf.placeholder(tf.float32, shape=(None,I), name="x")
x2 = tf.placeholder(tf.float32, shape=(None,H), name="x2")

y = tf.placeholder(tf.float32, shape=(None,1), name="y")

w1 = tf.get_variable("w1",shape=(I,H),dtype=tf.float32)
w2 = tf.get_variable("w2",shape=(H,O),dtype=tf.float32)

b1 = tf.get_variable("b1",shape=(H,),dtype=tf.float32)
b2 = tf.get_variable("b2",shape=(O,),dtype=tf.float32)

h1 = tf.nn.tanh(tf.nn.xw_plus_b(x, w1, b1))

h2_v1 = tf.nn.xw_plus_b(h1, w2, b2)
h2_v2 = tf.nn.xw_plus_b(x2, w2, b2)

cost_v1 = tf.nn.l2_loss(h2_v1 - y) # labels essentially become obsolete
cost_v2 = tf.nn.l2_loss(h2_v2 - y)

cost = cost_v1 + cost_v2

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    optimizer = tf.train.GradientDescentOptimizer(0.03)
    opt = optimizer.minimize(cost)
    #opt_grads = optimizer.compute_gradients(cost,var_list=[w2,b2])
    #optimizer.apply_gradients(opt_grads)
    
    #ph = sess.partial_run_setup([cost,opt],[h1])
    
    w1_old = w1.eval(sess)
    w2_old = w2.eval(sess)
    
    #pred_h1 = sess.run([h2],{h1:Z}) # seems to require values at input layer  
    loss, _ = sess.run([cost, opt],{x:X,x2:Z,y:Yx}) # in practice, I won't even need to enter in a value for Y
    w1_new = w1.eval(sess)
    w2_new = w2.eval(sess)
    
    # diff1 appears to be much larger than diff2
    diff1 = np.linalg.norm(w1_new - w1_old)
    diff2 = np.linalg.norm(w2_new - w2_old)
    
    halt= True