import numpy as np
from numpy.matrixlib.defmatrix import matrix
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import sgd
import sys

matplotlib.rc('legend', fontsize=18)

data_matrix = np.genfromtxt("helio_a.csv", delimiter=",")
data_x = np.genfromtxt("helio_x.csv", delimiter=",")
data_b = np.genfromtxt("helio_b.csv", delimiter=",")

def generate_noise(rows, distribution):
    result = np.zeros((rows,))
    for i in range(rows):
        result[i] = distribution(i)
    return result

def proj(matrix, split):
    (u, s, v) = np.linalg.svd(matrix)
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



datapoints = 212
dims = 100
A = data_matrix
x = data_x
b = data_b

iters = 212*60000
exp = 20
learning_rate = 0.01
delta = 0.1


for delta in [0.05, 0.1, 0.2]:
    for learning_rate in [0.005, 0.01, 0.02]:
        print("starting: " + str(delta) + ", " + str(learning_rate))
        noise_b = lambda x: np.random.normal(0,delta)
        exp_points = int(iters/(datapoints*100))
        xs_exact_loss = np.zeros((exp,exp_points+1))
        xs_noisy_loss = np.zeros((exp,exp_points+1))
        xs_exactm_loss = np.zeros((exp,exp_points+1))
        xs_noisym_loss = np.zeros((exp,exp_points+1))
        xs_exact = np.zeros((exp,exp_points+1,dims))
        xs_noisy = np.zeros((exp,exp_points+1,dims))
        xs_exactm = np.zeros((exp,exp_points+1,dims))
        xs_noisym = np.zeros((exp,exp_points+1,dims))
        AT = np.transpose(A)
        (xs_exact_n) = sgd.sgd(A, b, np.zeros((dims,)), learning_rate, iters, 21200)[1]
        for j in range(exp):
            sys.stdout.write("running: " + str(j) + '\r')
            sys.stdout.flush()
            bn = generate_noise(datapoints, noise_b)
            b2 = b + bn
            (xs_noisy_n) = sgd.sgd(A, b2, np.zeros((dims,)), learning_rate, iters, 21200)[1]
            xs_exact[j] = xs_exact_n
            xs_noisy[j] = xs_noisy_n
            for i in range(exp_points+1):
                xs_exactm[j,i] = np.mean(xs_exact[j,(i//2):i+1,:], axis=0)
                xs_noisym[j,i] = np.mean(xs_noisy[j,(i//2):i+1,:], axis=0)
            xs_exact_loss[j] = np.linalg.norm(np.dot(x-xs_exact[j],AT), axis=1)**2
            xs_noisy_loss[j] = np.linalg.norm(np.dot(x-xs_noisy[j],AT), axis=1)**2
            xs_exactm_loss[j] = np.linalg.norm(np.dot(x-xs_exactm[j],AT), axis=1)**2
            xs_noisym_loss[j] = np.linalg.norm(np.dot(x-xs_noisym[j],AT), axis=1)**2
        sys.stdout.write("completed\r\n")
        
        np.save('out_sgd/xs_exact_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_exact)
        np.save('out_sgd/xs_noisy_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_noisy)
        np.save('out_sgd/xs_exactm_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_exactm)
        np.save('out_sgd/xs_noisym_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_noisym)
        np.save('out_sgd/xs_exact_loss_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_exact_loss)
        np.save('out_sgd/xs_noisy_loss_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_noisy_loss)
        np.save('out_sgd/xs_exactm_loss_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_exactm_loss)
        np.save('out_sgd/xs_noisym_loss_'+str(delta)+'_'+str(learning_rate)+'.npy', xs_noisym_loss)
        
        mean_exact = np.mean(np.linalg.norm(x-xs_exact, axis=2)**2, axis=0)
        mean_noisy = np.mean(np.linalg.norm(x-xs_noisy, axis=2)**2, axis=0)
        std_exact = np.std(np.linalg.norm(x-xs_exact, axis=2)**2, axis=0)
        std_noisy = np.std(np.linalg.norm(x-xs_noisy, axis=2)**2, axis=0)
        mean_exactm = np.mean(np.linalg.norm(x-xs_exactm, axis=2)**2, axis=0)
        mean_noisym = np.mean(np.linalg.norm(x-xs_noisym, axis=2)**2, axis=0)
        std_exactm = np.std(np.linalg.norm(x-xs_exactm, axis=2)**2, axis=0)
        std_noisym = np.std(np.linalg.norm(x-xs_noisym, axis=2)**2, axis=0)
         
        mean_exact_weak = np.linalg.norm(data_x - np.mean(xs_exact, axis = 0), axis = 1)**2
        mean_noisy_weak = np.linalg.norm(data_x - np.mean(xs_noisy, axis = 0), axis = 1)**2
        
        mean_exactm_weak = np.linalg.norm(data_x - np.mean(xs_exactm, axis = 0), axis = 1)**2
        mean_noisym_weak = np.linalg.norm(data_x - np.mean(xs_noisym, axis = 0), axis = 1)**2

        np.savetxt("results/csv/xs_exact_loss"+str(delta)+"_"+str(learning_rate)+".csv", [np.mean(xs_exact_loss, axis=0)[[0,-1]], np.std(xs_exact_loss, axis=0)[[0,-1]]])
        np.savetxt("results/csv/xs_noisy_loss"+str(delta)+"_"+str(learning_rate)+".csv", [np.mean(xs_noisy_loss, axis=0)[[0,-1]], np.std(xs_noisy_loss, axis=0)[[0,-1]]])
        np.savetxt("results/csv/xs_exactm_loss"+str(delta)+"_"+str(learning_rate)+".csv", [np.mean(xs_exactm_loss, axis=0)[[0,-1]], np.std(xs_exactm_loss, axis=0)[[0,-1]]])
        np.savetxt("results/csv/xs_noisym_loss"+str(delta)+"_"+str(learning_rate)+".csv", [np.mean(xs_noisym_loss, axis=0)[[0,-1]], np.std(xs_noisym_loss, axis=0)[[0,-1]]])
        
        
        np.savetxt("results/csv/xs_exact_error"+str(delta)+"_"+str(learning_rate)+".csv", [mean_exact[[0,-1]], std_exact[[0,-1]]])
        np.savetxt("results/csv/xs_noisy_error"+str(delta)+"_"+str(learning_rate)+".csv", [mean_noisy[[0,-1]], std_noisy[[0,-1]]])
        np.savetxt("results/csv/xs_exactm_error"+str(delta)+"_"+str(learning_rate)+".csv", [mean_exactm[[0,-1]], std_exactm[[0,-1]]])
        np.savetxt("results/csv/xs_noisym_error"+str(delta)+"_"+str(learning_rate)+".csv", [mean_noisym[[0,-1]], std_noisym[[0,-1]]])
        
        np.savetxt("results/csv/xs_exact_weak"+str(delta)+"_"+str(learning_rate)+".csv", [mean_exact_weak[[0,-1]]])
        np.savetxt("results/csv/xs_noisy_weak"+str(delta)+"_"+str(learning_rate)+".csv", [mean_noisy_weak[[0,-1]]])
        np.savetxt("results/csv/xs_exactm_weak"+str(delta)+"_"+str(learning_rate)+".csv", [mean_exactm_weak[[0,-1]]])
        np.savetxt("results/csv/xs_noisym_weak"+str(delta)+"_"+str(learning_rate)+".csv", [mean_noisym_weak[[0,-1]]])
        
        if delta == 0.05:
            plt.errorbar([100*i for i in range(exp_points+1)],np.mean(xs_exact_loss,axis=0),np.std(xs_exact_loss,axis=0), fmt='g-', errorevery=60)
            plt.errorbar([100*i for i in range(exp_points+1)],np.mean(xs_exactm_loss,axis=0),np.std(xs_exact_loss,axis=0), fmt='y-', errorevery=60)
            if learning_rate == 0.005: plt.legend(['SGD loss', 'ASGD loss'])
            plt.savefig('results/loss_'+str(0)+'_'+str(learning_rate)+'.png')
            plt.close()

            plt.errorbar([100*i for i in range(exp_points+1)],mean_exact,yerr=std_exact, fmt='g-', errorevery=60)
            plt.errorbar([100*i for i in range(exp_points+1)],mean_exactm,yerr=std_exactm, fmt='y-', errorevery=60)
            if learning_rate == 0.005: plt.legend(['SGD error', 'ASGD error'])
            plt.savefig('results/error_'+str(0)+'_'+str(learning_rate)+'.png')
            plt.close()

        plt.errorbar([100*i for i in range(exp_points+1)],np.mean(xs_noisy_loss,axis=0),np.std(xs_noisy_loss,axis=0), fmt='g-', errorevery=60)
        plt.errorbar([100*i for i in range(exp_points+1)],np.mean(xs_noisym_loss,axis=0),np.std(xs_noisy_loss,axis=0), fmt='y-', errorevery=60)
        plt.savefig('results/loss_'+str(delta)+'_'+str(learning_rate)+'.png')
        plt.close()
        
        plt.errorbar([100*i for i in range(exp_points+1)],mean_noisy,yerr=std_noisy, fmt='g-', errorevery=60)
        plt.errorbar([100*i for i in range(exp_points+1)],mean_noisym,yerr=std_noisym, fmt='y-', errorevery=60)
        plt.savefig('results/error_'+str(delta)+'_'+str(learning_rate)+'.png')
        plt.close()

        plt.errorbar([100*i for i in range(exp_points+1)],mean_exact,yerr=std_exact, fmt='b-', errorevery=60)
        plt.plot([100*i for i in range(exp_points+1)],mean_exact_weak, 'y-')
        if delta == 0.05 and learning_rate == 0.005: plt.legend(['Weak error', 'Strong error'])
        plt.savefig('results/exact_weak_'+str(delta)+'_'+str(learning_rate)+'.png')
        
        plt.errorbar([100*i for i in range(exp_points+1)],mean_exactm,yerr=std_exactm, fmt='c-', errorevery=60)
        plt.plot([100*i for i in range(exp_points+1)],mean_exactm_weak, 'g-')
        if delta == 0.05 and learning_rate == 0.005: plt.legend(['SGD weak error','ASGD weak error', 'SGD strong error',  'ASGD strong error'])
        plt.savefig('results/mean_exact_weak_'+str(delta)+'_'+str(learning_rate)+'.png')
        plt.close()

        plt.errorbar([100*i for i in range(exp_points+1)],mean_noisy,yerr=std_noisy, fmt='b-', errorevery=60)
        plt.plot([100*i for i in range(exp_points+1)],mean_noisy_weak, 'y-')
        if delta == 0.05 and learning_rate == 0.005: plt.legend(['Weak error', 'Strong error'])
        plt.savefig('results/noisy_weak_'+str(delta)+'_'+str(learning_rate)+'.png')

        plt.errorbar([100*i for i in range(exp_points+1)],mean_noisym,yerr=std_noisym, fmt='c-', errorevery=60)
        plt.plot([100*i for i in range(exp_points+1)],mean_noisym_weak, 'g-')
        if delta == 0.05 and learning_rate == 0.005: plt.legend(['SGD weak error','ASGD weak error', 'SGD strong error',  'ASGD strong error'])
        plt.savefig('results/mean_noisy_weak_'+str(delta)+'_'+str(learning_rate)+'.png')
        plt.close()

        split = 5
        (l_p, h_p) = proj(data_matrix, split)
        low_error_exact =   np.log(np.mean((np.dot(l_p, np.reshape(data_x-xs_exact, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        low_error_noisy =   np.log(np.mean((np.dot(l_p, np.reshape(data_x-xs_noisy, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        low_error_exactm =  np.log(np.mean((np.dot(l_p, np.reshape(data_x-xs_exactm, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        low_error_noisym =  np.log(np.mean((np.dot(l_p, np.reshape(data_x-xs_noisym, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        
        high_error_exact =  np.log(np.mean((np.dot(h_p, np.reshape(data_x-xs_exact, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        high_error_noisy =  np.log(np.mean((np.dot(h_p, np.reshape(data_x-xs_noisy, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        high_error_exactm = np.log(np.mean((np.dot(h_p, np.reshape(data_x-xs_exactm, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        high_error_noisym = np.log(np.mean((np.dot(h_p, np.reshape(data_x-xs_noisym, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        
        low_std_exact =     np.log(np.std((np.dot(l_p, np.reshape(data_x-xs_exact, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        low_std_noisy =     np.log(np.std((np.dot(l_p, np.reshape(data_x-xs_noisy, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        low_std_exactm =    np.log(np.std((np.dot(l_p, np.reshape(data_x-xs_exactm, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        low_std_noisym =    np.log(np.std((np.dot(l_p, np.reshape(data_x-xs_noisym, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        
        high_std_exact =    np.log(np.std((np.dot(h_p, np.reshape(data_x-xs_exact, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        high_std_noisy =    np.log(np.std((np.dot(h_p, np.reshape(data_x-xs_noisy, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        high_std_exactm =   np.log(np.std((np.dot(h_p, np.reshape(data_x-xs_exactm, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))
        high_std_noisym =   np.log(np.std((np.dot(h_p, np.reshape(data_x-xs_noisym, (exp, exp_points+1, 100, 1))))**2, axis=(0,1,3)))

        plt.plot([100*i for i in range(exp_points+1)],low_error_exact,'b-')
        plt.plot([100*i for i in range(exp_points+1)],high_error_exact,'y-')
        plt.legend(['Low proj. error', 'High proj. error'])
        plt.savefig('results/exact_high_low_'+str(split) + "_" +str(delta)+'_'+str(learning_rate)+'.png')
        
        plt.plot([100*i for i in range(exp_points+1)],low_error_exactm,'c-')
        plt.plot([100*i for i in range(exp_points+1)],high_error_exactm,'g-')
        if delta == 0.05 and learning_rate == 0.005: plt.legend(['SGD low proj.', 'SGD high proj.', 'ASGD low proj.', 'ASGD high proj.'])
        plt.savefig('results/exactm_high_low_'+str(split) + "_" +str(delta)+'_'+str(learning_rate)+'.png')
        plt.close()

        plt.plot([100*i for i in range(exp_points+1)],low_error_noisy,'b-')
        plt.plot([100*i for i in range(exp_points+1)],high_error_noisy,'y-')
        if delta == 0.05 and learning_rate == 0.005: plt.legend(['Low projection (error', 'High proj. error'])
        plt.savefig('results/noisy_high_low_'+str(split) + "_" +str(delta)+'_'+str(learning_rate)+'.png')

        plt.plot([100*i for i in range(exp_points+1)],low_error_noisym,'c-')
        plt.plot([100*i for i in range(exp_points+1)],high_error_noisym,'g-')
        if delta == 0.05 and learning_rate == 0.005: plt.legend(['SGD low proj.', 'SGD high proj.', 'ASGD low proj.', 'ASGD high proj.'])
        plt.savefig('results/noisym_high_low_'+str(split) + "_" +str(delta)+'_'+str(learning_rate)+'.png')
        plt.close()


