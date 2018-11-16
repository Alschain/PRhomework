import os
import copy

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#loading data
dataPath = './data'
files = os.listdir(dataPath)

X = []

for eachfile in files:
    filepath = os.path.join(dataPath,eachfile)
    data = sio.loadmat(filepath)

    x = data['x']
    y = data['y']

    for i in range(len(x)):
        X.append([x[i][0],y[i][0]])

#convert to ndarray
X_matrix = np.array(X)

#Gauss Function, F(x)=P(x), args: Samples, mu, cov
#return the probability of all samples for gauss function
#used for calculating and adjusting tau
#return all samples' results after Gauss Function
def gauss_function(Y, mu, cov):
    gauss = multivariate_normal(mean=mu, cov=cov)
    return gauss.pdf(Y)


#Expectation, args: Samples, mu, cov, tau
#tau are used for the M-Step, return tau
def Expectation(X, mu, cov, tau):
    N = X.shape[0]
    K = tau.shape[0]

    T = np.zeros((N, K))

    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = gauss_function(X, mu[k], cov[k])

    for k in range(K):
        T[:, k] = tau[k] * prob[:, k]
    for i in range(N):
        T[i, :] /= np.sum(T[i, :])
    return T

#Maximization, args: Samples, tau
#calculate mu, new tau and cov matrix and return
def Maximization(X, T):
    Num, Dimension = X.shape
    K = T.shape[1]

    mu = np.zeros((K, Dimension))
    cov = []
    tau = np.zeros(K)

    for k in range(K):
        SumT = np.sum(T[:, k])
        for d in range(Dimension):
            mu[k, d] = np.sum(np.multiply(T[:, k], X[:, d])) / SumT
        cov_k = np.zeros((Dimension, Dimension))
        for i in range(Num):
            cov_k += T[i, k] * np.dot(np.expand_dims((X[i] - mu[k]),0).T , np.expand_dims((X[i] - mu[k]),0)) / SumT
        cov.append(cov_k)
        tau[k] = SumT / Num
    cov = np.array(cov)
    return mu, cov, tau


#EM-Algorithm, args: Samples, the number of gauss function and epsilon
#After the algorithm, return the mu, cov matrix and the tau
def EMAlgorithm(X, K, epsilon):
    N, Dimension = X.shape
    mu = np.random.rand(K, Dimension)
    cov = np.array([np.eye(Dimension)] * K)
    tau = np.array([1.0 / K] * K)
    count = 1
    while True:
        print(count)
        count += 1
        old_mu = copy.deepcopy(mu)
        T = Expectation(X, mu, cov, tau)
        mu, cov, tau = Maximization(X, T)
        if abs(np.linalg.norm(old_mu - mu)) < epsilon:
            print('Done!')
            break
    return mu, cov, tau


#Gaussian Mixed Model with two function, theta = (tau, mu1, mu2, sigma1, sigma2)
#tau means the probability of sample belonging to each gauss function
#mu1, mu2 are the mean value of each function
#sigma1, sigma2 are the cov matrix of each function
K = 2
Epsilon = 1e-8
mu, cov, tau = EMAlgorithm(X_matrix, K, Epsilon)

print('mu:',mu)
print('cov',cov)
print('tau',tau)


#print plot in 3D space
fig = plt.figure()
ax = Axes3D(fig)
plot_X = X_matrix[:, 0]
plot_Y = X_matrix[:, 1]

#make the class of sample witg the higher probability
#color different plots
Z = np.zeros((X_matrix.shape[0], K))
for k in range(K):
    Z[:, k] = gauss_function(X_matrix,mu[k],cov[k])

T = Expectation(X_matrix,mu,cov,tau)

plot_Z = np.zeros((X_matrix.shape[0],1))
colors = ['r' for i in range(plot_Z.shape[0])]
for each in range(Z.shape[0]):
    plot_Z[each] = Z[each][np.argmax(T[each])]
    colors[each] = 'r' if np.argmax(T[each]) == 0 else 'b'

plt.title("GMM")
ax.scatter(plot_X, plot_Y, plot_Z, color=colors)
ax.set_xlabel('Frequency')
ax.set_ylabel('Standard Deviation')
ax.set_zlabel('Probability')
plt.show()