# -*- coding: utf-8 -*-
"""
@author: Andrew St.Germain
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt('ex1data2.txt',delimiter=',')

y = data[:,len(data[0])-1]
X = data[:,0:len(data[0])-1]
m = len(y)
numberOfFeatures = len(X[0])

# Plot 3D data
"""fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.show()"""

# Plot 2D data
#plt.plot(X,y,'x')


def normalizeFeatures(X):
    normX = X
    for i in range(numberOfFeatures):
        column = X[:,i]
        normX[:,i] = (column - (sum(column)/len(column)))/(max(column) - min(column))
    return normX
  
def computeCost(X,y,Theta):
    summation = 0;
    for i in range(m):
        summation += (np.dot(Theta,X[i]) - y[i])**2
    J = (1.0/(2.0*m))*summation
    return J

def gradientDescent(X,y,Theta,alpha,iterations):
    Jvector = np.zeros(iterations)
    newTheta = Theta
    for n in range(iterations):
        ThetaTemp = newTheta
        for j in range(numberOfFeatures+1):
            summation = 0
            for i in range(m):
                summation += (np.dot(X[i],ThetaTemp) - y[i])*X[i,j]
            newTheta[j] -= (alpha/m)*(summation)
        Jvector[n] = computeCost(X,y,newTheta)
        #print "Theta = " + str(newTheta)
        #print "J = " + str(Jvector[n])
    return [newTheta,Jvector]

def runGradientDescentMethod2D(X):
    normX = X      
    #normX = normalizeFeatures(X)
    X = np.ones([m,numberOfFeatures+1])
    X[:,1:] = normX
    #y = (y - (sum(y)/len(y)))/(max(y) - min(y))

    alpha = 0.01
    iterations = 1500

    Theta = np.zeros(numberOfFeatures+1)

    GD = gradientDescent(X,y,Theta,alpha,iterations)
    Theta = GD[0]
    #print GD[1][-1]

    h = np.zeros(m)
    for i in range(m):
        h[i] = np.dot(X[i],Theta)
    # Plot data and hypothesis for one feature
    plt.hold(True)
    plt.plot(X[:,1],y,'x')
    plt.plot(X[:,1],h)
    plt.hold(False)

    # Plot data and hypothesis for two features
    """fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2])
    Axes3D.plot_surface(X[:,1],X[:,2],h)"""
 
   
def runGradientDescentMethod3D(X):
    normX = X      
    #normX = normalizeFeatures(X)
    X = np.ones([m,numberOfFeatures+1])
    X[:,1:] = normX
    #y = (y - (sum(y)/len(y)))/(max(y) - min(y))

    alpha = 0.01
    iterations = 1500

    Theta = np.zeros(numberOfFeatures+1)

    GD = gradientDescent(X,y,Theta,alpha,iterations)
    Theta = GD[0]
    #print GD[1][-1]

    h = np.zeros(m)
    for i in range(m):
        h[i] = np.dot(X[i],Theta)
    # Plot data and hypothesis for one feature
    plt.hold(True)
    plt.plot(X[:,1],y,'x')
    plt.plot(X[:,1],h)
    plt.hold(False)

    # Plot data and hypothesis for two features
    """fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2])
    Axes3D.plot_surface(X[:,1],X[:,2],h)"""
    

def runNormalEquationMethod2D(X):
    normX = X      
    #normX = normalizeFeatures(X)
    X = np.ones([m,numberOfFeatures+1])
    X[:,1:] = normX
    XTXinv = np.linalg.pinv(np.transpose(np.matrix(X))*np.matrix(X))
    Theta = XTXinv*np.transpose(np.matrix(X))*np.transpose(np.matrix(y))
    h = np.zeros(m)
    for i in range(m):
        h[i] = np.dot(X[i],Theta)

    # Plot data and hypothesis for one feature
    plt.hold(True)
    plt.plot(X[:,1],y,'x')
    plt.plot(X[:,1],h)
    plt.hold(False)
    

def runNormalEquationMethod3D(X):
    normX = X      
    #normX = normalizeFeatures(X)
    X = np.ones([m,numberOfFeatures+1])
    X[:,1:] = normX
    XTXinv = np.linalg.pinv(np.transpose(np.matrix(X))*np.matrix(X))
    Theta = XTXinv*np.transpose(np.matrix(X))*np.transpose(np.matrix(y))
    h = np.zeros(m)
    for i in range(m):
        h[i] = np.dot(X[i],Theta)

    # Plot data and hypothesis for one feature
    plt.hold(True)
    plt.plot(X[:,1],y,'x')
    plt.plot(X[:,1],h)
    plt.hold(False)
    

#runGradientDescentMethod2D(X)
#runNormalEquationMethod2D(X)
    
    
    
    