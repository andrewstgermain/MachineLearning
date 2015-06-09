# -*- coding: utf-8 -*-
"""
@author: Andrew
"""

""" import all necessary libraries """

import numpy as np
import matplotlib.pyplot as plt
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
def costFunction(Theta,X,y):
    summation = 0
    for i in range(m):
        summation += -y[0,i]*np.log(sigmoid(np.dot(X[i,:],np.transpose(Theta))))-(1-y[0,i])*np.log(1-sigmoid(np.dot(X[i,:],np.transpose(Theta))))
    J = (1.0/m)*summation
    return J

# Computes the gradient of the cost function for each theta in a matrix of
# thetas with featrues X and outputs y
def gradient(Theta,X,y):
    grad = np.matrix(np.zeros(n+1))
    for j in range(n+1):
        summation = 0
        for i in range(m):
            summation += (sigmoid(np.dot(X[i,:],np.transpose(Theta))) - y[0,i])*X[i,j]
        grad[0,j] = (1.0/m)*summation
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

# Predicts the results (positive or negative) of a set of data features after
# finding an optimal Theta from a training set
def predict(Theta,X):
    predictions = np.zeros(m)
    for i in range(m):
        probability = sigmoid(np.dot(X[i,:],np.transpose(Theta)))
        if probability >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions
        
 
""" The main part of the program """  
 
# Get the data from the file
data = np.loadtxt('ex2data1.txt',delimiter=',')

# Get the features (X) and the outputs (y) from the data
X = np.matrix(data[:,0:len(data[0])-1])
y = np.matrix(data[:,len(data[0])-1]) 

# Plot the data to see what it looks like
plt.hold(True) # This makes sure the points are all on the same plot
plotData(X,y)
plt.hold(False)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.title("+ = Admitted, o = Not Admitted")
plt.show()

# m is number of data points
m = np.size(y)
# n is number of features
n = np.size(X[0])

# Set up the features matrix (add a column of ones as the first column)
newX = np.matrix(np.ones([m, n+1]))
newX[:,1:n+1] = X
X = newX

# Initialize theta
Theta = np.matrix(np.zeros(n + 1))
J = float(costFunction(Theta,X,y))
print "Initial Theta = " + str(Theta)
print "Initial Cost = " + str(J)

# Use the equalivalent of the fminunc function in Matlab
Result = op.minimize(fun = costFunction, x0 = Theta, args = (X, y), method = 'TNC', jac = gradient)
# Set Theta to the final,optimized value
Theta = Result.x
J = float(costFunction(Theta,X,y))
print "Final Theta = " + str(Theta)
print "Final Cost = " + str(J)

# Plot the Decision Boundary
plt.hold(True) # This makes sure the points are all on the same plot
plotDecisionBoundary(Theta,X,y)
plt.hold(False)
plt.axis([30, 100, 30, 100])
plt.title("+=Admitted, o=Not Admitted, --=Decision Boundary")
plt.show()

# Check the prediction function with the training set to check accuracy
p = predict(Theta,X)
print "Prediction Accuracy with training set: " + str(np.mean(p == y) * 100.0) + "%"













