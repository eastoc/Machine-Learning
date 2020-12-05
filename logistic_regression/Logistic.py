#Time: 2020.10
#Python Version: 3.8
#logistic regression
#Author: Wu Fangdong(East)
#Gradient_Descent
#Database from Machine Learning for Zhou Zhihua
# -*- coding: UTF-8
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

class logistic:

    def sigmoid(x):
        z = 1 / (1 + np.exp(-x))
        return z

    def cost_function(X,y,w):
        temp = np.dot(X,w)
        temp = logistic.sigmoid(temp)
        J = y*np.log(temp)+(1-y)*np.log(1-temp)#极大似然估计
        return J.sum()

    def Grad(X,y,w):
        temp = np.dot(X,w)
        temp = y-logistic.sigmoid(temp)
        gra = np.dot(X.T,temp)
        gra = gra.reshape(-1,1)
        return gra

    def draw_loss(loss):
        iteration = range(len(loss))
        print(len(iteration))
        plt.title('Loss')
        plt.xlabel('Iterations')
        plt.ylabel('loss')
        plt.plot(iteration,loss)
        plt.show()

    def Upgrade_GD(X,y,w,learning_rate,Iterations):
        loss=[]
        for i in range(Iterations):
            grad = logistic.Grad(X,y,w)
            w = w + learning_rate * grad
            #print loss
            loss_temp = logistic.cost_function(X,y,w)
            loss = np.append(loss,loss_temp)
            if (i % 10 == 0):
                print('The iterations is ',i,' the loss is ',loss_temp)
        print(len(loss))
        logistic.draw_loss(loss)
        return w

    def Upgrade_BGD(X,y,w,learning_rate):
        m = len(y)
        batch = 3
        index = np.random.sample(range(1,m),batch)
        print(index)

    def Upgrade_SGD(X,y,w,learning_rate):
        m = len(y)
        index = np.random.randint(1, m, size=1)
        print(index)
        X_SGD = X[index, :]
        y_SGD = y[index]

    def Adam(X,y,w,learning_rate):
        pass

    def Init(n):#基于标准正态分布的参数初始化
        w = np.random.normal(0,1,n)
        w = w.reshape(-1,1)
        print(w)
        return w

    def Xavier_Init(n):#Xavier参数初始化
        pass

    def train(X,y,method,Iterations = 100,learning_rate = 0.5):
        m,n = X.shape
        w = logistic.Init(n)
        print('Learning rate is ',learning_rate)
        if method == 'GD':
            print('Optimalization method is gradient descent')
            return logistic.Upgrade_GD(X,y,w,learning_rate,Iterations)
        elif method == 'BGD':
            print('Optimalization method is batch gradient descent')
            return logistic.Upgrade_BGD(X,y,w,learning_rate,Iterations)
        elif method == 'SGD':
            print('Optimalization method is stochatic gradient descent')
            return logistic.Upgrade_SGD(X,y,w,learning_rate,Iterations)

    def test(X,w):
        y = logistic.sigmoid(np.dot(X,w))
        if (y >= 0.5):
            y = 1
            print("It's a good xigua")
        elif y < 0.5:
            y = 0


