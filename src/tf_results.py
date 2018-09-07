import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import numpy as np
import sys


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

settings = [(15, 0.005)]


epochs = 81
exp = 5
finals = np.zeros((exp,7960))

x_final = np.reshape(np.load('trained_network.npy'), [7960,])
u = np.load('hessian_u.npy')
s = np.load('hessian_s.npy')
v = np.load('hessian_v.npy')
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999, 2499, 3999, 5499, 6999]:
    plt.plot(range(7960), v[i],'b-')
    plt.savefig('tfresults/v'+str(i+1)+".png")
    plt.close()

    first_layer = np.reshape(v[i,:7840], (28,28,5,2))
    result = np.zeros((28*2, 28*5))
    for j in range(28*2):
        for k in range(28*5):
            result[j,k] = first_layer[j%28, k%28, k//28, j//28]
    plt.imshow(result, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('tfresults/layer_1_v_'+str(i)+'.png', bbox_inches='tight')
    plt.close()

    result = np.zeros((12,10))
    result = np.reshape(v[i,7840:], (12,10))
    plt.imshow(result, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('tfresults/other_v_'+str(i)+'.png', bbox_inches='tight')
    plt.close()

tu = np.load('trained_hessian_u.npy')
ts = np.load('trained_hessian_s.npy')
tv = np.load('trained_hessian_v.npy')
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 999, 2499, 3999, 5499, 6999]:

    first_layer = np.reshape(tv[i,:7840], (28,28,5,2))
    result = np.zeros((28*2, 28*5))
    for j in range(28*2):
        for k in range(28*5):
            result[j,k] = first_layer[j%28, k%28, k//28, j//28]
    plt.imshow(result, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('tfresults/trained_layer_1_v_'+str(i)+'.png', bbox_inches='tight')
    plt.close()

    result = np.zeros((10,12))
    result = np.reshape(tv[i,7840:], (10,12))
    plt.imshow(result, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('tfresults/trained_other_v_'+str(i)+'.png', bbox_inches='tight')
    plt.close()



(low_proj, high_proj) = proj(u,s,v, 10)
low_dist = np.zeros((exp,epochs))
high_dist = np.zeros((exp,epochs))
low_dist_average = np.zeros((exp,epochs))
high_dist_average = np.zeros((exp,epochs))
strong = np.zeros((exp, epochs))
strong_avg = np.zeros((exp,epochs))

offset = 21
for i in range(offset,offset+exp):
    print(i)
    iters = np.load('out_data/full_iter'+str(i)+'.npy')
    final = np.load('out_data/final'+str(i)+'.npy')
    finals[i-offset,:] = final
    low_result = np.dot(low_proj, final)
    high_result = np.dot(high_proj, final)

    average_values = np.zeros((epochs, 7960))
    average_values[0] = iters[0]
    for j in range(1,epochs):
        average_values[j] = (j*average_values[j-1])/(j+1)+iters[j]/(j+1)

    low_dist[i-offset,:] = np.asarray([(np.linalg.norm(np.dot(low_proj,value)-low_result)**2) for value in iters])
    high_dist[i-offset,:] = np.asarray([(np.linalg.norm(np.dot(high_proj,value)-high_result)**2) for value in iters])
    
    low_dist_average[i-offset,:] = np.asarray([(np.linalg.norm(np.dot(low_proj,value)-low_result)**2) for value in average_values])
    high_dist_average[i-offset,:] = np.asarray([(np.linalg.norm(np.dot(high_proj,value)-high_result)**2) for value in average_values])

    strong[i-offset,:] = np.asarray([(np.linalg.norm(value-final)**2) for value in iters])
    strong_avg[i-offset,:] = np.asarray([(np.linalg.norm(value-final)**2) for value in average_values])


plt.errorbar(range(epochs), np.mean(low_dist,axis=0), yerr= np.std(low_dist,axis=0), fmt='b-')
plt.errorbar(range(epochs), np.mean(high_dist,axis=0), yerr=np.std(high_dist,axis=0), fmt='c-')
plt.errorbar(range(epochs), np.mean(low_dist_average,axis=0), yerr= np.std(low_dist_average,axis=0), fmt='y-')
plt.errorbar(range(epochs), np.mean(high_dist_average,axis=0), yerr=np.std(high_dist_average,axis=0), fmt='g-')
plt.legend(['SGD Low Frequency Projection Error', 'SGD High Frequency Projection Error', 'Average SGD Low Frequency Projection Error', 'Average SGD High Frequency Projection Error'])
plt.savefig('tfresults/high_low_20exp.png')
plt.close()

plt.plot(range(epochs), np.log10(np.mean(low_dist,axis=0)), 'b-')
plt.plot(range(epochs), np.log10(np.mean(high_dist,axis=0)), 'c-')
plt.plot(range(epochs), np.log10(np.mean(low_dist_average,axis=0)), 'y-')
plt.plot(range(epochs), np.log10(np.mean(high_dist_average,axis=0)), 'g-')
plt.legend(['SGD Low Frequency Projection Error', 'SGD High Frequency Projection Error', 'Average SGD Low Frequency Projection Error', 'Average SGD High Frequency Projection Error'])
plt.savefig('tfresults/high_low_20exp_log.png')
plt.close()

plt.errorbar(range(epochs),np.mean(strong, axis=0),yerr=np.std(strong, axis=0), fmt='b-')
plt.errorbar(range(epochs),np.mean(strong_avg, axis=0),yerr=np.std(strong_avg, axis=0), fmt='g-')
plt.legend(['SGD Strong Error', 'Averaged SGD Strong Error'])
plt.savefig('nn_error.png')
plt.close()

colors = ['b','g','r','c','y']
for i in range(5):
    plt.plot(range(epochs), low_dist[i,:], colors[i]+'-')
    plt.plot(range(epochs), high_dist[i,:], colors[i]+'--')
plt.savefig('tfresults/5_error.png')
plt.close()


colors = ['b','g','r','c','y']
for i in range(5):
#    plt.figure(figsize=(200,200))
    plt.plot(range(500), (finals[i,5000:5500]), '-')
plt.savefig('tfresults/final_trained.png')
plt.close()

plt.plot(range(7960), np.std(finals, axis=0), 'b-')
plt.savefig('tfresults/final_trained_stddev.png')
plt.close()

