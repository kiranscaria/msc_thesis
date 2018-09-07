import numpy as np
from numpy.matrixlib.defmatrix import matrix
from typing import Callable
import matplotlib.pyplot as plt
import sgd
import util
from mnist import MNIST

class linear_layer():
    def __init__(self, rows, cols, learning_rate):
        self.f_in = rows
        self.f_out = cols
        self.learning_rate = learning_rate
        self.initialize()
    
    def initialize(self):
        self.iter = 0
        self.weights = np.random.normal(size=(self.f_in, self.f_out), scale=0.1)
        self.b = np.random.normal(size=(self.f_out,), scale=0.1)
        self.cache = None
        self.b_avg = np.zeros((self.f_out,))
        self.w_avg = np.zeros((self.f_in, self.f_out))

    def forwards(self, in_data, avg = False):
        assert(self.cache == None)
        if(avg):
            result = np.dot(in_data, self.w_avg) + self.b_avg
        else:
            result = np.dot(in_data, self.weights) + self.b
        self.cache = in_data
        return result

    def backwards(self, out_gradient):
        assert(not (self.cache is None))
        self.iter += 1
        in_gradient = np.dot(out_gradient, np.transpose(self.weights))
        b_delta = self.learning_rate*np.sum(out_gradient, axis=0)/(self.cache.shape[0])
        w_delta = self.learning_rate*np.dot(np.transpose(self.cache), out_gradient)/(self.cache.shape[0])
        self.b -= b_delta
        self.weights -= w_delta
        
        self.w_avg = self.w_avg*(self.iter - 1)/(self.iter) + self.weights/self.iter
        self.b_avg = self.b_avg*(self.iter - 1)/(self.iter) + self.b/self.iter

        self.cache = None
        return in_gradient
    
    def clear_cache(self):
        self.cache = None

class sigmoid_layer():
    def __init__(self):
        self.cache = None
    def initialize(self):
        return

    def forwards(self, in_data, avg = False):
        e = np.exp(in_data)
        result = e/(1+e)
        self.cache = result
        return result

    def backwards(self, out_gradient):
        assert(not (self.cache is None))
        in_gradient = out_gradient*self.cache*(1-self.cache)
        self.cache = None
        return in_gradient
    
    def clear_cache(self):
        self.cache = None

class relu_layer():
    def __init__(self):
        self.cache = None

    def initialize(self):
        return
    def forwards(self, in_data, avg = False):
        result = np.maximum(in_data, 0)
        self.cache = np.greater(in_data, 0.0)
        return result

    def backwards(self, out_gradient):
        assert(not (self.cache is None))
        in_gradient = out_gradient*(self.cache.astype(int))
        self.cache = None
        return in_gradient
    
    def clear_cache(self):
        self.cache = None
 

class loss_layer():
    def __init__(self):
        self.cache = None

    def initialize(self):
        return
    def forwards(self, in_data, targets, avg = False):
        e = np.exp(in_data)
        out = e/(np.sum(e, axis=1, keepdims = True))
        self.cache = out - targets 
        result = targets*np.log(out)

        return -np.sum(result)

    def backwards(self):
        assert(not (self.cache is None))
        result = self.cache 
        self.cache = None
        return result 

    def clear_cache(self):
        self.cache = None

class ff_nn():
    def __init__(self, layers):
        self.layers = layers
    def initialize(self):
        [layer.initialize() for layer in self.layers]

    def forwards(self, in_data, targets, avg = False):
        data = in_data
        for layer in self.layers[:-1]:
            data = layer.forwards(data, avg)
        data = self.layers[-1].forwards(data, targets)
        return data
    def backwards(self):
        back_gradient = self.layers[-1].backwards()
        for layer in reversed(self.layers[:-1]):
            back_gradient = layer.backwards(back_gradient)
    def clear_cache(self):
        [layer.clear_cache() for layer in self.layers]

layer0 = linear_layer(784,100,0.005)
#layer0 = linear_layer(784,100,0.01)
layer1 = relu_layer()
layer2 = linear_layer(100,10,0.005)
#layer2 = linear_layer(100,10,0.01)
layer3 = loss_layer()

nn = ff_nn([layer0, layer1, layer2, layer3])

mndata = MNIST('/home/ubuntu/msc_thesis/')
data, labels = mndata.load_training()
data = np.asarray(data)/256.0


labels_onehot = np.zeros((len(data),10))
for i in range(len(labels)):
    labels_onehot[i,labels[i]] = 1.0


exp = 20

iters = 40
loss = np.zeros((exp, iters+1))
mean_loss = np.zeros((exp, iters+1))
for e in range(exp):
    j = 0
    nn.initialize()
    for i in range(iters*20000):
        if i % (20000) == 0:
            loss[e, j]= (nn.forwards(data, labels_onehot)/data.shape[0])
            nn.clear_cache()
     
            mean_loss[e, j] = (nn.forwards(data, labels_onehot, avg = True)/data.shape[0])
            nn.clear_cache()
            print(loss[e,j],mean_loss[e,j])
            j+=1
        index = np.random.randint(data.shape[0])
        nn.forwards(data[[index],:], labels_onehot[[index],:])
        nn.backwards()
    loss[e, j]= (nn.forwards(data, labels_onehot)/data.shape[0])
    nn.clear_cache()

    mean_loss[e, j] = (nn.forwards(data, labels_onehot, avg = True)/data.shape[0])
    nn.clear_cache()
    print(loss[e,j])
    print('finished experiment ' + str(e))

print(np.std(loss, axis=0))
plt.errorbar(range(iters+1),np.mean(loss, axis=0),yerr=np.std(loss, axis=0), fmt='b-', errorevery=5)
plt.errorbar(range(iters+1),np.mean(mean_loss, axis=0),yerr=np.std(mean_loss, axis=0), fmt='g-', errorevery=5)
plt.legend(['SGD Loss', 'Averaged SGD Loss'])
plt.savefig('nn_loss_0.01.png')
plt.close()
