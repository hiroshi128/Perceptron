#coding:utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#read data from file and return as numpy data array
def readData(filename):
    data = np.loadtxt(filename)
    return(data)

#set data label 0 to -1
def setNegativeLabel(y):
    y[y==0]=-1
    return y

#separate data     
def dataSplit(data):
    X_train_test, X_val, y_train_test, y_val = train_test_split(data[:, 0:2], data[:, 2], test_size=0.2, random_state= 42) #separate data into 20% and 80%
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size = 0.25, random_state = 42) #separate again remaining data. 80% * 0.25 = 20%
    return (X_train, X_val, X_test, y_train, y_val, y_test)

#implementation of perceptron alogrithm
def perceptron(X, y, X_val=None, y_val=None, eta=0.01, early_stop = False):
    X = np.column_stack((np.ones(X.shape[0]), X)) # insert 1s into each training data
    np.random.seed(42)
    weight = np.random.rand(X.shape[1])
    w_opt = weight
    num_epoch = 1000
    min_E_in = float("inf")
    current_E_in = float("inf")
    min_E_val = float("inf")
    current_E_val = float("inf")
    min_E_val_epoch = 0
    E_in = []
    E_val = []
    E_in.append(evalError(X, y, w_opt))
    if(X_val is not None and y_val is not None):
        X_val = np.column_stack((np.ones(X_val.shape[0]), X_val)) # insert 1s into each training data
        E_val.append(evalError(X_val, y_val, w_opt))

    for i in range(num_epoch):
        error = 0.0
        for j in range(len(X)):
            y_prediction = sign(np.dot(weight, X[j]))
            if(y_prediction != y[j]):
                weight = weight + eta * y[j] * X[j]
                current_E_in = evalError(X, y, weight)
                if(current_E_in < min_E_in):
                    w_opt = weight
                    min_E_in = current_E_in 
        E_in.append(min_E_in)
        if(X_val is not None and y_val is not None):
            current_E_val = evalError(X_val, y_val, w_opt)
            E_val.append(current_E_val)
            if(current_E_val < min_E_val):
                min_E_val = current_E_val
                min_E_val_epoch = i
            elif(early_stop and current_E_val >= min_E_val * 1.05): #Terminate when validation error hikes 5%
                break

    plotErrorGraph(E_in, E_val, eta, early_stop)
    return w_opt

#plot E_in and E_val
def plotErrorGraph(E_in, E_val, learning_rate, early_stop):
    plt.title("learning rate = "+str(learning_rate)+", Early stop = "+str(early_stop))
    plt.plot(E_in, '-', label="Training Error")
    plt.plot(E_val, '--', label = "Validation Error")
    plt.legend()
    plt.show()

#calculate mean squared error
def evalError(X, y, w):
    error = 0.0
    for i in range(len(X)):
        y_pred = sign(np.dot(w, X[i]))
        if(y[i] != y_pred):
            error = error + pow((y[i] - y_pred), 2)
    error = error / len(X)
    return error

#positive input or zero=> returns 1, negative input=> returns -1
def sign(val):
    if(val >= 0):
        return 1
    else:
        return -1

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('Usage: $python perceptron.py <data.dat>')
    else:
        data = readData(args[1])
        data[:,2] = setNegativeLabel(data[:,2])
        X_train, X_val, X_test, y_train, y_val, y_test = dataSplit(data)

        #step 4~6. Try different learning rates
        learning_rates = [0.001,0.0005,0.0001,0.00005,0.00001]
        for eta in learning_rates:
            perceptron(X_train, y_train, X_val, y_val, eta, early_stop = False)

        #step 7. Try termination condition
        perceptron(X_train, y_train, X_val, y_val, 0.0001, early_stop = True)

        #step 8. Final Hypothesis and Generalization Error
        # I chose eta as 0.0001 by the trial of different learning rates.
        # get the final hypothesis with validation data and training data.
        w_final = perceptron(np.r_[X_train, X_val], np.r_[y_train, y_val], eta = 0.0001, early_stop = True)

        X_test = np.column_stack((np.ones(X_test.shape[0]), X_test)) # insert 1s into test examples to measure error
        print("Generalization Error: ", evalError(X_test, y_test, w_final)) #calculate the generalization error with the final hypothesis