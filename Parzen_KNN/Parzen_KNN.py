import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#Generate Gauss Data, args:mu, sigma and sample number
#return samples
def generate_data(mu, sigma, sampleNum):
    return np.random.multivariate_normal(mu, sigma, sampleNum)

#Generate test data, args:gauss data, test number
#number of points are testNum ** 2
#use data to get the test data, max and min,average test number
#return test
def generate_test(data, testNum):
    test = []
    #make sure N=1 have samples
    X = np.linspace(min(data[:,0]),max(data[:,0]),testNum) if data.shape[0] != 1 else np.linspace(min(data[:,0])-1,max(data[:,0])+1,testNum)
    Y = np.linspace(min(data[:,1]),max(data[:,1]),testNum) if data.shape[0] != 1 else np.linspace(min(data[:,1])-1,max(data[:,1])+1,testNum)
    test.append(X)
    test.append(Y)
    return np.array(test)

#Gauss Function, F(x)=P(x), args: Samples, mu, cov
#return the probability of all samples for gauss function
#used for calculating and adjusting tau
#return all samples' results after Gauss Function
def gauss_function(samples, mu, cov):
    gauss = multivariate_normal(mean=mu, cov=cov)
    return gauss.pdf(samples)

#phi function, get gauss value, args:x, xi, hn, mu, sigma
#return gauss value
def phi(x, xi, hn, mu, sigma):
    return gauss_function((x - xi) / hn, mu, sigma) / ((hn * hn) * 1.0)

#Parzen windows, get prob estimate, args:data, test, h, mu, sigma
#return probs
def parzen_window(data, test, h, mu, sigma):
    dataNum = data.shape[0]
    size = np.array([test.shape[1],test.shape[1]])
    probs = np.zeros(size)
    hn = h / math.sqrt(dataNum)
    for i in range(size[0]):
        for j in range(size[1]):
            tempTest = np.array([[test[0][i],test[1][j]]]).repeat(dataNum,axis=0)
            tempProb = phi(tempTest, data, hn, mu, sigma)
            probs[i][j] = sum(tempProb) / dataNum if dataNum != 1 else tempProb
    return probs

#KNN, args:data, test and K neighbor
#return probs
def KNN(data, test, k):
    dataNum = data.shape[0]
    size = np.array([test.shape[1],test.shape[1]])
    probs = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            tempTest = np.array([[test[0][i],test[1][j]]]).repeat(dataNum,axis=0)
            tempProb = np.linalg.norm(tempTest - data, axis=1)
            tempProb = sorted(tempProb) if dataNum != 1 else tempProb
            r = tempProb[int(k) - 1] if dataNum != 1 else tempProb
            V = math.pi * (r ** 2)
            probs[i][j] = k * 1.0 / dataNum / V
    return probs

h1 = [0.25, 1, 4]
sampleNum = [1, 16, 256, 10000]
testNum = 50
mu = np.zeros(2)
sigma = np.eye(2)

for N in sampleNum:
    #Data generate
    data = generate_data(mu, sigma, N)
    test = generate_test(data, testNum)

    #Parzen Window
    for h in h1:
        print('Parzen Window for h=%.2f, n=%d' % (h, N))
        probs = parzen_window(data, test, h, mu, sigma)
        xplot, yplot = np.meshgrid(test[0], test[1])

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.title('Parzen window for h=%.2f, n=%d' % (h, N))
        ax.plot_surface(xplot, yplot, probs, cmap='coolwarm')
        plt.show()
    
    #KNN
    k = math.sqrt(N)
    print('KNN for k=%d'%(k))
    probs = KNN(data, test, k)
    xplot, yplot = np.meshgrid(test[0], test[1])

    fig = plt.figure()
    ax = Axes3D(fig)
    plt.title('KNN for k=%d' % (k))
    ax.plot_surface(xplot, yplot, probs, cmap='coolwarm')
    plt.show()
    