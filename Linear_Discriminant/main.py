import matplotlib.pyplot as plt
import numpy as np
import Load

data_path = 'data/西瓜.csv'
data,label = Load.loadFile(data_path)
data_pos = []
data_neg = []
#draw scatter figure
for i in range(len(label)):
    if label[i] == 1:
        plt.scatter(data[i,0],data[i,1],s=20,c='r')
        data_pos=np.append(data_pos,data[i,:])
    elif label[i]  == 0 :
        plt.scatter(data[i,0],data[i,1],s=20,c='b')
        data_neg = np.append(data_neg,data[i,:])
plt.title('Property Scatter Figure')
plt.show()

data_pos = data_pos.reshape(-1,2)
data_neg = data_neg.reshape(-1,2)

u_pos = data_pos.mean(axis = 0)
u_neg = data_neg.mean(axis = 0)
u_pos = u_pos.T
u_neg = u_neg.T

cov_pos = np.cov(data_pos.T)
cov_neg = np.cov(data_neg.T)

#within-class scatter matrix
S_w=cov_pos+cov_neg
w = np.dot(np.linalg.inv(S_w),u_pos-u_neg)
print(w)

x_pos_new = np.dot(data_pos, w)
print(x_pos_new)
x_neg_new = np.dot(data_neg, w)
y_pos_new = [1 for i in range(len(x_pos_new))]
y_neg_new = [2 for i in range(len(x_neg_new))]


plt.plot(x_pos_new, y_pos_new, 'b*')
plt.plot(x_neg_new, y_neg_new, 'ro')
plt.show()

