import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import sys

#Code adapted from UCL/Deepmind module COMPGI22

log_period_samples = 20000
batch_size = 1

def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_placeholders():
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  return x, y_

def proj(u,s,v, split):
    #(u, s, v) = np.linalg.svd(matrix)
    m = u.shape[1]
    n = v.shape[0]
    s2 = np.zeros((m,n))
    for i in range(min(m,n)):
        s2[i,i] = s[i]
    low_proj = np.zeros((n,n))
    high_proj = np.zeros((n,n))
    for i in range(min(m,n)):
        if i < split:
             low_proj[i,:] = v[i,:]
        else:
             high_proj[i,:] = v[i,:]
    return (np.dot(u, np.dot(s2, low_proj)), np.dot(u, np.dot(s2, high_proj)))

x_final = np.reshape(np.load('trained_network.npy'), [7960,])
u = np.load('hessian_u.npy')
s = np.load('hessian_s.npy')
v = np.load('hessian_v.npy')
(low_proj, high_proj) = proj(u,s,v, 10)


num_epochs = 80
learning_rate = 0.005

tf.reset_default_graph()
x, y_ = get_placeholders()
mnist = get_data() 
subset = np.random.choice(range((mnist.train.images).shape[0]),10000,replace=False)
train_subset_images = mnist.train.images[subset]
train_subset_labels = mnist.train.labels[subset]

all_var = tf.get_variable("all", [796,10])
w1 = all_var[:784,:]
b1 = all_var[784:785,:]
w2 = all_var[785:795,:]
b2 = all_var[795:,:]
o1 = tf.nn.relu(tf.tensordot(x,w1,1)+b1)
o2 = tf.tensordot(o1,w2,1)+b2
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=o2))
g = tf.gradients(loss, [w1,w2,b1,b2])

lr = tf.constant(learning_rate)
sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update = sgd_optimizer.minimize(loss)
y_index = tf.argmax(y_,1)
o_index = tf.argmax(o2,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_index, o_index),tf.float32))

hessian = tf.hessians(loss,all_var) 

# Train.
i, train_accuracy, test_accuracy = 0, [], []
params = []
log_period_updates = int(log_period_samples / batch_size)
with tf.train.MonitoredSession() as sess:
  while i//log_period_updates <= num_epochs:
    
    # Update.
    i += 1
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(update, feed_dict={x:batch_xs,y_: batch_ys})
    if i % log_period_updates == 0:
      train_accuracy_v = sess.run(accuracy, feed_dict={x:mnist.train.images, y_:mnist.train.labels})
      test_accuracy_v = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})

      train_accuracy.append(train_accuracy_v)
      
      values = np.reshape(sess.run(all_var), [7960,])
      params.append(values)
      train_accuracy_v
    if i == 1:
      hessian_result = sess.run(hessian, feed_dict={x:train_subset_images, y_:train_subset_labels})
      hessian_mat = np.reshape(hessian_result[0],[7960, 7960])
      (u, s, v) = np.linalg.svd(hessian_mat)
      plt.plot(range(s.size),np.log10(s), 'b-')
      plt.savefig('tfresults/singular_values_'+ str(i) +'.png')
      plt.close()
  hessian_result = sess.run(hessian, feed_dict={x:train_subset_images, y_:train_subset_labels})
  param_result = sess.run(all_var)
  param_result = np.reshape(param_result, [7960,])
  hessian_mat = np.reshape(hessian_result[0],[7960, 7960])
  (u, s, v) = np.linalg.svd(hessian_mat)
  (low_proj, high_proj) = proj(u,s,v, 10)

  plt.plot(range(s.size), np.log10(s), 'b-')
  plt.savefig('tfresults/singular_values_'+ str(i) +'.png')
  plt.close()

  low_result = np.dot(low_proj, param_result)
  high_result = np.dot(high_proj, param_result)
  low_dist = [(np.linalg.norm(np.dot(low_proj,value)-low_result)**2) for value in params]
  high_dist = [(np.linalg.norm(np.dot(high_proj,value)-high_result)**2) for value in params]
  plt.plot(range(len(low_dist)), low_dist, 'b-')
  plt.plot(range(len(low_dist)), high_dist, 'r-')
  plt.legend(['Low Projection Norm', 'High Projection Norm'])
  plt.savefig('tfresults/high_low_10.png')
  plt.close()
  if (len(sys.argv) > 1):
      n = int(sys.argv[1])
      np.save('out_data/full_iter' + str(n) + '.npy', np.asarray(params))
      np.save('out_data/final' + str(n) + '.npy', (param_result))
  np.save('trained_hessian_u.npy', u)
  np.save('trained_hessian_s.npy', s)
  np.save('trained_hessian_v.npy', v)
