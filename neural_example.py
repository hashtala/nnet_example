# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:06:14 2019

@author: gela
"""



from Nafo import artificial_neural_net as ANN
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt




X1 = np.random.randn(100,2) + np.array([2,2])
X2 = np.random.randn(100,2) + np.array([-2,-2])
X3 = np.random.randn(100,2) + np.array([4,-4])
X4 = np.random.randn(100,2) + np.array([-4,4])
X5 = np.random.randn(100,2) + np.array([10,-10])
X6 = np.random.randn(100,2) + np.array([20,-10])
X7 = np.random.randn(100,2) + np.array([-6,-8])
X8 = np.random.randn(100,2) + np.array([10,19])


X = np.vstack([X1,X2, X3, X4, X5, X6, X7, X8]).astype(np.float32)

Y = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100 +[5]*100 + [6]*100 + [7]*100).astype(np.int32)

Model_Neural = ANN.Artificial_Neural_Net([(2, 100, T.nnet.relu),
                                    (100, 100, T.nnet.relu),
                                    (100, 8, T.nnet.relu),
                                    (1, 1, T.nnet.softmax)]
                                    )


prediction, error_rate = Model_Neural.fit(X,Y, mu = 0.99)

plt.scatter(X1[:,0], X1[:,1])
plt.scatter(X2[:,0], X2[:,1])
plt.scatter(X3[:,0], X3[:,1])
plt.scatter(X4[:,0], X4[:,1])
plt.scatter(X5[:,0], X5[:,1])
plt.scatter(X6[:,0], X6[:,1])
plt.scatter(X7[:,0], X7[:,1])
plt.scatter(X8[:,0], X8[:,1])
plt.show()
plt.plot(error_rate)








