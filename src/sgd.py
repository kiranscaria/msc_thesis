import numpy as np
import util

def sgd(A, b, x0, lr, iterations, xs_every):
    (rows,cols) = A.shape
    x = x0
    xs = np.zeros((int(iterations/xs_every)+1, cols))
    xs[0] = x
    j = 1
    for i in range(1,iterations+1):
        index = np.random.randint(rows)
        x -= lr*(np.dot(A[index,:], x) - b[index])*A[index,:]#(np.reshape(A[index, :],(cols,)))
        if(i % xs_every == 0):
            xs[j] = x
            j += 1
    return (x, xs)

def mean_sgd(A, b, x0, lr, iterations, xs_every):
    (rows,cols) = A.shape
    x = x0
    xs = np.zeros((int(iterations/xs_every)+1, cols))
    xs[0] = x
    j = 1
    m = x
    for i in range(1,iterations+1):
        index = np.random.randint(rows)
        x -= lr*(np.dot(A[index,:], x) - b[index])*A[index,:]#(np.reshape(A[index, :],(cols,)))
        m = m*((i-1)/(i)) + x/i
        if(i % xs_every == 0):
            xs[j] = m
            j += 1
    return (m, xs)

def forwards(A, x):
    result = [A]
    layer_in = A
    for layer in x:
        layer_out = util.sigmoid(np.dot(layer_in, layer))
        result += [layer_out]
        layer_in = layer_out
    return result

def loss(A, b, x):
    result = [A]
    layer_in = A
    for layer in x:
        layer_out = util.sigmoid(np.dot(layer_in, layer))
        layer_in = layer_out
    return np.sum(np.linalg.norm(layer_out-b,axis=1))

def backwards(outputs, x, b, lr):
    out_gradient = (outputs[-1]-b)
    result = []
    for (layer, input_v, out) in zip(x[::-1], outputs[-2::-1], outputs[:0:-1]):
        out_grad = out_gradient*out*(1-out)
        in_gradient = np.dot(out_grad,np.transpose(layer))
        layer = layer - lr*np.outer(input_v,out_grad)
        out_gradient = in_gradient
        result = [layer] + result
    return result
    

def sgd_nn(A, b, x0, lr, iterations):
    (rows,cols) = A.shape
    x = x0
    xs = [None]*(iterations+1)
    xs[0] = loss(A, b, x)
    for i in range(iterations):
        index = np.random.randint(rows)
        outputs = forwards(A[index,:], x)
        x = backwards(outputs, x, b[index,:], lr)
        xs[i+1] = loss(A, b, x)
    return (x, xs)
