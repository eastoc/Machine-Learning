#Load train database

import numpy as np
import matplotlib.pyplot as plt
import csv
def loadFile(data_path,mode = False):

    dataset = np.genfromtxt(data_path, delimiter=',')

    m,n = np.shape(dataset)

    data = dataset[1:m,1:3]
    label = dataset[1:m, 3]

    label = label.astype(int)
    label = label.reshape(-1,1)

    return data,label

