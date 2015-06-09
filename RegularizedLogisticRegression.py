# -*- coding: utf-8 -*-
"""
@author: Andrew
"""

""" import all necessary libraries """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as op


""" Define all of the functions needed to run the program """

# Plots the data for two features in matrix X and binary y outputs of 0 or 1
def plotData(X,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    # plot the positive data points (y = 1)
    plt.plot(X[pos[1],0], X[pos[1],1], marker = '+', c = 'k')
    # plot the negative data points (y = 0)
    plt.plot(X[neg[1],0], X[neg[1],1], marker = 'o', c = 'y')

# Calculates the Sigmoid function for a given z value/matrix
def sigmoid(z):
    g = 1.0/(1+np.exp(-z))
    return g

# Computes the value of the cost function for a certain matrix of theta values
# with features X and outputs y
def regularizedCostFunction(Theta,X,y,Lambda):
    Theta = np.matrix(Theta)
    J = (1.0/m)*(-y*np.log(sigmoid(X*np.transpose(Theta)))-(1-y*np.log(1-sigmoid(X*np.transpose(Theta))))) + (Lambda/(2.0*m))*np.sum(np.square(Theta[0,1:np.size(Theta)+1]))
    return float(J)

# Computes the gradient of the cost function for each theta in a matrix of
# thetas with featrues X and outputs y
def regularizedGradient(Theta,X,y,Lambda):
    Theta = np.matrix(Theta)
    grad = np.matrix(np.zeros(n+1))
    grad[0,0] = (1.0/m)*(np.transpose(sigmoid(X*np.transpose(Theta))) - y)*X[:,0]
    grad[0,1:n+2] = (1.0/m)*(np.transpose(sigmoid(X*np.transpose(Theta))) - y)*X[:,0] + (float(Lambda)/m)*Theta[0,1:n+2]
    return grad

# Plots the decision boundary for given Theta values of a two feature matrix X 
# with outputs in matrix y
def plotDecisionBoundary(Theta,X,y):
    plotData(X[:,1:(n+1)],y)
    dimensions = np.shape(X)
    # Make sure there are only two features (2D graph)
    if dimensions[1] <= 3:
        # Choose the two end points two define the line
        plot_x = np.array([float(min(X[:,2]))-2,  float(max(X[:,2]))+2])
        # Calculate the decision boundary line
        plot_y = (-1.0/Theta[2])*(Theta[1]*plot_x + Theta[0])
        # Plot the decision boundary now
        plt.plot(plot_x, plot_y)
    else:
        print "Matrix X has two many features for a 2D plot."

def predict(Theta,X):
    predictions = np.zeros(m)
    for i in range(m):
        probability = sigmoid(np.dot(X[i,:],np.transpose(Theta)))
        if probability >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions

def regularizedNumberOfFeatures(degree):
    d = 1
    n = 0
    while d<=degree:
        n += d+1
        d += 1
    return n
    
def featureMap(X1,X2,degree):
    X1 = np.array(np.transpose(X1))
    X2 = np.array(np.transpose(X2))
    m = np.size(X1)
    n = regularizedNumberOfFeatures(degree)
    out = np.ones([m,n+1])
    for i in range(1,degree+1):
        for j in range(i):
            for column in range(1,n):
                out[:,column] = np.matrix(np.power(X1,(i-j))*np.power(X2,j))
    return np.matrix(out)


""" The main part of the program """  

# Get the data from the file
data = np.loadtxt('ex2data2.txt',delimiter=',')

# Get the features (X) and the outputs (y) from the data
X = np.matrix(data[:,0:len(data[0])-1])
y = np.matrix(data[:,len(data[0])-1]) 

# Plot the data to see what it looks like
plt.hold(True) # This makes sure the points are all on the same plot
plotData(X,y)
plt.hold(False)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title("+ = Admitted, o = Not Admitted")
plt.show()

# m is number of data points
m = np.size(y)
# n is number of features
degree = 6
n = regularizedNumberOfFeatures(degree)
Lambda = 1

X = featureMap(X[:,0],X[:,1],degree)

# Initialize theta
Theta = np.matrix(np.zeros(n+1))

print regularizedCostFunction(Theta,X,y,Lambda)
print np.shape(regularizedGradient(Theta,X,y,Lambda))

# Use the equalivalent of the fminunc function in Matlab
Result = op.minimize(fun = regularizedCostFunction, x0 = Theta, args = (X, y, Lambda), method = 'TNC', jac = regularizedGradient)
# Set Theta to the final,optimized value
print Result










