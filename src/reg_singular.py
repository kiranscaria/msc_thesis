import numpy as np
from numpy.matrixlib.defmatrix import matrix
from typing import Callable
import matplotlib.pyplot as plt
import sgd

data_matrix = np.genfromtxt("helio_a.csv", delimiter=",")
data_x = np.genfromtxt("helio_x.csv", delimiter=",")
data_b = np.genfromtxt("helio_b.csv", delimiter=",")
(u,s,v) = np.linalg.svd(data_matrix)
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 39, 59, 79, 99]:
    plt.plot(range(100), v[i],'b-')
    plt.savefig('reg_sin_vec/v'+str(i+1)+".png")
    plt.close()

