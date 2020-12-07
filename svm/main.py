#SVM for linear kernel and Gauss kernel
#Author：Wu Fangdong
#Time：20.12.7
#python version: 3.8

from libsvm.svm import *
from libsvm.svmutil import *
import Load
from pylab import *

path = 'data/西瓜.csv'
X,y= Load.loadFile(path)
print(y.shape)

# # #train for SVM
prob = svm_problem(y, X)
param_1 = svm_parameter('-t 0 -c 4 -b 1')
param_2 = svm_parameter('-t 2 -c 4 -b 1')

model_1 = svm_train(prob,param_1)
model_2 = svm_train(prob,param_2)

svm_save_model('model_1_file', model_1)
svm_save_model('model_2_file', model_2)

# # # plot a scatter figgure
figure(figsize=(10,6), dpi=80)
for x,flag in zip(X,y):
    if flag == 1:
        plot(x[0],x[1],'o',color='black')
    elif flag == 0:
        plot(x[0],x[1],'x',color='blue')
show()