# -*- coding: utf-8 -*-
"""
@author: Andrew
"""

import numpy as np
import pygame as pg
from pygame.locals import *
import PIL
import matplotlib.pyplot as plt

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

# Displays the data as the digits that they are   
def visualizeData(data):
    #answers = data[:,0]
    #print answers
    #data = data[:,1:785]
    numberOfDigits = np.size(data[:,0])
    for digit in range(numberOfDigits):
        pixels = data[digit,:]
        image = pixels.reshape([28,28])
        plt.imshow(image, interpolation='nearest')
        plt.gray()
        plt.show()

# Calculates the Sigmoid function for a given z value/matrix
def sigmoid(z):
    g = 1.0/(1+np.exp(-z))
    return g

# Predicts the output values for a set of data given theta values
def predict(Theta1, Theta2, X):
    # m is the number of examples
    m = np.size(X[:,0])
    # n is the number of features (pixels)
    n = np.size(X[0,:])
    
    a1 = np.matrix(np.ones([m,n+1]))
    a1[:,1:n+1] = X
    h1 = sigmoid(a1 * np.transpose(Theta1))
    a2 = np.matrix(np.ones([m,np.size(h1[0,:])+1]))
    a2[:,1:np.size(h1[0,:])+1] = h1
    h2 = sigmoid(a2 * np.transpose(Theta2))
    p = np.argmax(h2, axis = 1)
    return p

# Gets the Theta values needed for the predict function
def getTheta():
    dataFile = open("Theta.txt","r")
    data = dataFile.read()
    data = np.loadtxt("Theta.txt")
    return data

# Number of features in the input layer (only can be 784 in this case)
inputLayerSize = 784
# Choose the number of features to have in the hidden layer (can be changed)
hiddenLayerSize = 25
# K is the number of labels output (10 for 10 digits)
K = 10

Theta = getTheta()
print Theta
Theta1 = np.reshape(Theta[0:hiddenLayerSize * (inputLayerSize + 1)], [hiddenLayerSize, (inputLayerSize + 1)])
Theta2 = np.reshape(Theta[(hiddenLayerSize * (inputLayerSize + 1)):np.size(Theta)+1], [K, (hiddenLayerSize + 1)])     
"""
data = getData('train.csv',4200)
y = data[:,0]
X = data[:,1:np.size(data[0,:])+1]
p = predict(Theta1,Theta2,X)

correct = (p == y)

print str((np.sum(correct)/float(np.size(y)))*100)+ " % correct for training data"

data = getData('train.csv',8400)
y = data[4200:8401,0]
X = data[4200:8401,1:np.size(data[0,:])+1]
p = predict(Theta1,Theta2,X)

correct = (p == y)

print str((np.sum(correct)/float(np.size(y)))*100)+ " % correct for test data"
"""
pg.init()
screen = pg.display.set_mode([240,240])
white = (255,255,255)
black = (0,0,0)
g50 = (50,50,50)
g100 = (100,100,100)
g150 = (150,150,150)
g200 = (200,200,200)
screen.fill(white)
# initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
myfont = pg.font.SysFont("monospace", 15)

# render text

while True:
    for event in pg.event.get():
        if event.type == QUIT:
            pg.quit()
        elif event.type == pg.MOUSEMOTION:
            endPos = pg.mouse.get_pos()
            if pg.mouse.get_pressed()==(1,0,0):
                pg.draw.line(screen,g200,startPos,endPos,29)
                pg.draw.line(screen,g150,startPos,endPos,23)
                pg.draw.line(screen,g100,startPos,endPos,17)
                pg.draw.line(screen,g50,startPos,endPos,11)
                pg.draw.line(screen,black,startPos,endPos,5)
            startPos=endPos
        pressed=pg.key.get_pressed()
        if pressed[K_RETURN]:
            pg.image.save(screen,'image.png')
            image = PIL.Image.open("image.png").convert("L")
            image = image.resize((28,28))
            number = np.array(image)
            x = 255 - number.reshape([1,784])
            visualizeData(x)
            p = predict(Theta1,Theta2,x)
            p = str(int(p))
            label = myfont.render(p, 1, black, white)
            screen.blit(label, (220, 220))
            print p
        if pressed[K_c]:
            screen.fill(white)            
        pg.display.update()