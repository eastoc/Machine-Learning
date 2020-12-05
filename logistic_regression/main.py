#Time: 2020.10
#Python Version: 3.8
#logistic regression
#Author: Wu Fangdong(East)
#Database from Machine Learning for Zhou Zhihua
# -*- coding: UTF-8

import numpy as np
import Load
from matplotlib import pyplot as plt
import Logistic as LR

data_path = 'data/西瓜.csv'
data,label = Load.loadFile(data_path)
w = LR.logistic.train(data,label,'GD',Iterations = 1000,learning_rate = 0.05)
print(w)
