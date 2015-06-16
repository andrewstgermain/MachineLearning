# -*- coding: utf-8 -*-
"""
@author: Andrew
"""

""" Import all necessary libraries """

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


""" Define all of the function needed to run the program """

# Gets the data from a file in the correct format (train.csv or test.csv), only
# getting a certain amount of digits dependent on how large you want the
# data set to be (max is 42000)
# Data is a matrix, where each row corresponds to a digit and the first column
# is the correct identification of that digit, and the rest of the columns are
# the pixels of that digit
def getData(fileName,numberOfDigits):
    # Open the file
    dataFile = open(fileName,'r')
    # Read the first two lines (skipping the first since it is only labels)
    line = dataFile.readline()
    line = dataFile.readline()
    # Strip each line of 'new line' marker and split at commas
    line = line.rstrip('\n')
    splitline = line.split(',')
    # Get an amount of data small enough for my computer
    data = np.matrix(np.zeros([numberOfDigits,len(splitline)]))
    for i in range(numberOfDigits):
        for j in range(len(splitline)):
            data[i,j] = float(splitline[j])
        line = dataFile.readline()
        line = line.rstrip('\n')
        splitline = line.split(',')
    # Close the file we were working with
    dataFile.close()
    return data
    
# Gets all of the data in a list instead of a matrix
def getDataList(fileName):
    train = open(fileName,'r')
    # Read lines(did it twice to skip first line of non-numbers)
    line = train.readline()
    line = train.readline()
    line = line.rstrip('\n')
    splitline = line.split(',')
    i = 0
    #length = 42000
    data = []
    while len(line) > 0:
        datavector = []
        for j in range(len(splitline)):
            datavector.append(float(splitline[j]))
        data.append(datavector)
        line = train.readline()
        line = line.rstrip('\n')
        splitline = line.split(',')
        i += 1
    train.close()
    return data
 
# Displays the data as the digits that they are   
def visualizeData(data):
    #answers = data[:,0]
    #print answers
    data = data[:,1:785]
    numberOfDigits = np.size(data[:,0])
    for digit in range(numberOfDigits):
        pixels = data[digit,:]
        image = pixels.reshape([28,28])
        plt.imshow(image, interpolation='nearest')
        plt.gray()
        plt.show()

# Randomly initializes the theta values for the neural network base on the
# number of incoming connections and outgoing connections
def initializeTheta(numberIn,numberOut):
    epsilon = 0.12
    Theta = np.random.rand(numberOut, 1 + numberIn) * 2 * epsilon - epsilon
    return Theta

# Calculates the Sigmoid function for a given z value/matrix
def sigmoid(z):
    g = 1.0/(1+np.exp(-z))
    return g
    
# Calculates the gradient of the Sigmoid function for a give z value/matrix
def sigmoidGradient(z):
    g = np.multiply(sigmoid(z),(1-sigmoid(z)))
    return g

# Calculates the cost function given certain Theta values, X features,
# y outputs, Lambda value, and number of labels (K)
# Theta1, Theta2, X, and y should all be matrices
def costFunction(Theta,X,y,Lambda,K,inputLayerSize,hiddenLayerSize):
    Theta1 = np.reshape(Theta[0:hiddenLayerSize * (inputLayerSize + 1)], [hiddenLayerSize, (inputLayerSize + 1)])
    Theta2 = np.reshape(Theta[(hiddenLayerSize * (inputLayerSize + 1)):np.size(Theta)+1], [K, (hiddenLayerSize + 1)]) 
    # m is the number of examples
    m = np.size(y)
    # n is the number of features (pixels)
    n = np.size(X[0,:])
    # Make y into a matrix with K labels output for each example
    Y = np.zeros([m,K])
    for i in range(m):
        # Only the y(i)th value in the output labels for each example is 1 
        Y[i,int(y[i])] = 1
    a1 = np.matrix(np.zeros([m,n+1]))
    a1[:,0] = np.matrix(np.ones([m,1]))
    a1[:,1:n+2] = X
    z2 = a1*np.transpose(Theta1)
    a2 = np.matrix(np.zeros([m,np.size(z2[0,:])+1]))
    a2[:,0] = np.matrix(np.ones([m,1]))
    a2[:,1:np.size(z2[0,:])+2] = sigmoid(z2)
    z3 = a2*np.transpose(Theta2)
    a3 = sigmoid(z3)
    J = np.sum((1.0/m)*(np.multiply(-Y,np.log(a3)) - np.multiply((1-Y),np.log(1-a3)))) + (Lambda/(2.0*m))*(np.sum(np.square(Theta1[:,1:np.size(Theta1[0,:])+1])) + np.sum(np.square(Theta2[:,1:np.size(Theta2[0,:])+1])))
    return J

# Calculates the gradient of the cost function
def costGradient(Theta,X,y,Lambda,K,inputLayerSize,hiddenLayerSize):
    Theta1 = np.reshape(Theta[0:hiddenLayerSize * (inputLayerSize + 1)], [hiddenLayerSize, (inputLayerSize + 1)])
    Theta2 = np.reshape(Theta[(hiddenLayerSize * (inputLayerSize + 1)):np.size(Theta)+1], [K, (hiddenLayerSize + 1)]) 
    # m is the number of examples
    m = np.size(y)
    # n is the number of features (pixels)
    n = np.size(X[0,:])
    # Make y into a matrix with K labels output for each example
    Y = np.zeros([m,K])
    for i in range(m):
        # Only the y(i)th value in the output labels for each example is 1 
        Y[i,int(y[i])] = 1
    
    Delta1 = 0;
    Delta2 = 0;
    for t in range(m):
        # Step 1
        a1 = np.matrix(np.zeros([1,n+1]))
        a1[:,0] = 1
        a1[:,1:n+2] = X[t,:]
        z2 = a1*np.transpose(Theta1)
        a2 = np.matrix(np.zeros([1,np.size(z2[0,:])+1]))
        a2[:,0] = 1
        a2[:,1:np.size(z2[0,:])+2] = sigmoid(z2)
        z3 = a2*np.transpose(Theta2)
        a3 = sigmoid(z3)
        # Step 2
        delta3 = a3 - Y[t,:]
        # Step 3
        delta2 = np.multiply((delta3*Theta2[:,1:np.size(Theta2[0,:])+1]),sigmoidGradient(z2))
        #print "delta3 = " + str(delta3)
        #print "delta2 = " + str(delta2)
        # Step 4
        Delta1 = Delta1 + np.transpose(delta2)*a1
        Delta2 = Delta2 + np.transpose(delta3)*a2
    #print np.shape(Delta1)
    #print np.shape(Delta2)
    #print Delta1
    #print Delta2
    # Regularize each Delta term into a D matrix
    D1 = np.matrix(np.zeros(np.shape(Delta1)))
    D2 = np.matrix(np.zeros(np.shape(Delta2)))
    D1[:,0] = (1.0/m)*Delta1[:,0]
    D1[:,1:np.size(D1[0,:])+1] = (1.0/m)*Delta1[:,1:np.size(Delta1[0,:])+1] + (Lambda/float(m))*Theta1[:,1:np.size(Theta1[0,:])+1]
    D2[:,0] = (1.0/m)*Delta2[:,0]    
    D2[:,1:np.size(D2[0,:])+1] = (1.0/m)*Delta2[:,1:np.size(Delta2[0,:])+1] + (Lambda/float(m))*Theta2[:,1:np.size(Theta2[0,:])+1]    
    # Unroll each gradient term into one grad matrix, which is just one row    
    grad = np.matrix(np.zeros([1,np.size(D1)+np.size(D2)]))
    grad[:,0:np.size(D1)] = np.reshape(D1,np.size(D1))
    grad[:,np.size(D1):np.size(grad)+1] = np.reshape(D2,np.size(D2))
    return grad

data = getData('train.csv',100)
#visualizeData(data)

# y is the actual value of each digit
y = data[:,0]
# X is the input of features (pixels) for each digit
X = data[:,1:np.size(data[0,:])+1]

# Choose the number of features to have in the hidden layer
numberFeaturesHidden = 25
# K is the number of labels output (10 for 10 digits)
K = 10
# Lambda is the regularization constant
Lambda = 0
inputLayerSize = 784
hiddenLayerSize = 25

# Theta1 is the matrix of theta weight values between the input layer and
# the hidden layer, and Theta2 is between the hidden layer and output layer
Theta1 = initializeTheta(np.size(X[0,:]),numberFeaturesHidden)
Theta2 = initializeTheta(numberFeaturesHidden,K)

Theta = np.zeros([1,np.size(Theta1)+np.size(Theta2)])
print np.size(Theta[:,0:np.size(Theta1)])
Theta[:,0:np.size(Theta1)] = np.reshape(Theta1,np.size(Theta1))
Theta[:,np.size(Theta1):np.size(Theta)+1] = np.reshape(Theta2,np.size(Theta2))

print np.size(Theta1)
print np.size(Theta2)

#J = costFunction(Theta,X,y,Lambda,K,inputLayerSize,hiddenLayerSize)
#grad = costGradient(Theta,X,y,Lambda,K,inputLayerSize,hiddenLayerSize)
#print J
#print grad
Result = op.minimize(fun = costFunction, x0 = Theta, args = (X,y,Lambda,K,inputLayerSize,hiddenLayerSize), method = 'TNC', jac = costGradient)

print Result







