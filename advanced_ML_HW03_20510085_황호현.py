#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt

def wkNN(Xtr,ytr,Xts,k,random_state=None):

    y_pred = []
    
    for t_point in Xts:
        dist = pairwise_distances([t_point],Xtr)[0]
        n_idx = np.argsort(dist)[:k]
        
        if len(np.unique(ytr)) == 3 :
            weight = {0:0, 1:0, 2:0}
        else :
            weight = {0:0, 1:0}
    
        for idx in n_idx:
            if dist[n_idx[-1]] == dist[n_idx[0]]:
                weight[ytr[idx]] = 1
            else:
                weight[ytr[idx]] = (dist[n_idx[-1]]-dist[idx])/(dist[n_idx[-1]]-dist[n_idx[0]]) + weight[ytr[idx]]

        y_pred.append([i for i in weight.keys() if weight[i] == max(weight.values())][0])
    
    return y_pred

def PNN(Xtr,ytr,Xts,k,random_state=None):

    y_pred = []
    labeling_class = {}
    
    num_class = len(np.unique(ytr))
    
    for cls in range(0,num_class):
        labeling_class[cls] = Xtr[ytr==cls]
        
    for datas in Xts:
        
        weight = {}
        
        for cls in labeling_class:
            weight[cls] = np.sort(pairwise_distances(labeling_class[cls], [datas]).reshape(1,-1)[0])[0:k]
            
            for i in range(1, k+1):
                weight[cls][i-1] = weight[cls][i-1]/i
        
        sum_labeling_class = {}
        for cls in labeling_class:
            sum_labeling_class[cls] = sum(weight[cls])
   
        y_pred.append([i for i in sum_labeling_class.keys() if sum_labeling_class[i] == min(sum_labeling_class.values())][0])

    return y_pred
   

X1,y1=datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=13)
Xtr1,Xts1, ytr1, yts1=train_test_split(X1,y1,test_size=0.2, random_state=22)

X2,y2=datasets.make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=3, n_classes=2, n_clusters_per_class=2, flip_y=0.2,random_state=75)
Xtr2,Xts2, ytr2, yts2=train_test_split(X2,y2,test_size=0.2, random_state=78)


# In[44]:


def cal_acc(pred, test_label):
    
    count_correct = sum(pred == test_label)
    
    return round(count_correct/len(pred), 4)


# In[39]:


klist=[3,5,7,9,11]

WKNN1=[]
Pnn1=[]

start = time.time()

for k in klist:
    wknn_1 = wkNN(Xtr1, ytr1, Xts1, k)
    pnn_1 = PNN(Xtr1, ytr1, Xts1, k)
    wknn_acc = cal_acc(wknn_1, yts1)
    pnn_acc = cal_acc(pnn_1, yts1)
    
    WKNN1.append(wknn_acc)
    Pnn1.append(pnn_acc)
        
        
during_time = time.time()-start

print("Elapsed time: ", during_time)
print('-----------------------------------------')
print('k          wkNN          PNN')
print('-----------------------------------------')
print(klist[0],'         ',WKNN1[0],'        ',Pnn1[0])
print(klist[1],'         ',WKNN1[1],'        ',Pnn1[1])
print(klist[2],'         ',WKNN1[2],'        ',Pnn1[2])
print(klist[3],'         ',WKNN1[3],'        ',Pnn1[3])
print(klist[4],'         ',WKNN1[4],'        ',Pnn1[4])
print('-----------------------------------------')


# In[64]:


k=7 # default

wkn_TF1=(wkNN(Xtr1, ytr1, Xts1, k)==yts1)
pn_TF1=(PNN(Xtr1, ytr1, Xts1, k)==yts1)

plt.xlim(-5,5)
plt.ylim(-5,5)

train_x=[]
train_y=[]
test_x=[]
test_y=[]

wkn_TF_x1=[]
wkn_TF_y1=[]

pn_TF_x1=[]
pn_TF_y1=[]

train_class = {}
test_class = {}

for cls in range(0,3) :
    train_class[cls] = Xtr1[ytr1==cls]
    test_class[cls] = Xts1[yts1==cls]
    
for tr_x, tr_y in Xtr1:
    train_x.append(tr_x)
    train_y.append(tr_y)

for tst_x, tst_y in Xts1:
    test_x.append(tst_x)
    test_y.append(tst_y)
    
for idx in range(len(wkn_TF1)):
    if wkn_TF1[idx] == True:
        continue
    else:
        wkn_TF_x1.append(train_x[idx])
        wkn_TF_y1.append(train_y[idx])

for idx in range(len(pn_TF1)):
    if pn_TF1[idx] == True:
        continue
    else:
        pn_TF_x1.append(train_x[idx])
        pn_TF_y1.append(train_y[idx])


plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(train_x, train_y, c='teal', marker='o',s=10)
plt.scatter(test_x, test_y, c='teal', marker='x',s=10)
plt.scatter(wkn_TF_x1, wkn_TF_y1, c='none', marker='s', edgecolors='red', s=30)
plt.scatter(pn_TF_x1, pn_TF_y1, c='none', marker='d', edgecolors='blue', s=30)
plt.legend(['Train','Test','Misclassifed by wkNN','Miscalssified by PNN'], loc='lower right')
plt.scatter(train_class[0][: ,0], train_class[0][:,1], color='purple', marker='o',s=10)
plt.scatter(train_class[1][: ,0], train_class[1][:,1], color='teal', marker='o',s=10)
plt.scatter(train_class[2][: ,0], train_class[2][:,1], color='yellow', marker='o',s=10)
            
plt.scatter(test_class[0][: ,0], test_class[0][:,1], color='purple', marker='x',s=10)
plt.scatter(test_class[1][: ,0], test_class[1][:,1], color='teal', marker='x',s=10)
plt.scatter(test_class[2][: ,0], test_class[2][:,1], color='yellow', marker='x',s=10)


# In[59]:


klist = [3,5,7,9,11]
WKNN2 = []
Pnn2 = []

start = time.time()

for k in klist:
    wknn_2 = wkNN(Xtr2, ytr2, Xts2, k)
    pnn_2 = PNN(Xtr2, ytr2, Xts2, k)
    wknn_acc = cal_acc(wknn_2, yts2)
    pnn_acc = cal_acc(pnn_2, yts2)
    
    WKNN2.append(wknn_acc)
    Pnn2.append(pnn_acc)
        
during_time = time.time()-start

print("Elapsed time: ", during_time)
print('-----------------------------------------')
print('k          wkNN          PNN')
print('-----------------------------------------')
print(klist[0],'         ',WKNN2[0],'        ',Pnn2[0])
print(klist[1],'         ',WKNN2[1],'        ',Pnn2[1])
print(klist[2],'         ',WKNN2[2],'        ',Pnn2[2])
print(klist[3],'         ',WKNN2[3],'        ',Pnn2[3])
print(klist[4],'         ',WKNN2[4],'        ',Pnn2[4])
print('-----------------------------------------')


# In[65]:


k=7
wkn_TF2=(wkNN(Xtr2, ytr2, Xts2, k)==yts2)
pn_TF2=(PNN(Xtr2, ytr2, Xts2, k)==yts2)
plt.xlim(-3,3)
plt.ylim(-5,5)

train_x=[]
train_y=[]
test_x=[]
test_y=[]

wkn_TF_x2=[]
wkn_TF_y2=[]

pn_TF_x2=[]
pn_TF_y2=[]

train_class = {}
test_class = {}

for cls in range(0,3) :
    train_class[cls] = Xtr2[ytr2==cls]
    test_class[cls] = Xts2[yts2==cls]
    
for tr_x, tr_y,_1,_2,_3,_4 in Xtr2:
    train_x.append(tr_x)
    train_y.append(tr_y)

for tst_x, tst_y,_1,_2,_3,_4 in Xts2:
    test_x.append(tst_x)
    test_y.append(tst_y)
    
for ind in range(len(wkn_TF2)):
    if wkn_TF2[idx] == True:
        continue
    else:
        wkn_TF_x2.append(train_x[idx])
        wkn_TF_y2.append(train_y[idx])

for idx in range(len(pn_TF2)):
    if pn_TF2[idx] == True:
        continue
    else:
        pn_TF_x2.append(train_x[idx])
        pn_TF_y2.append(train_y[idx])


plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(train_x, train_y, c='teal', marker='o',s=10)
plt.scatter(test_x, test_y, c='teal', marker='x',s=10)
plt.scatter(wkn_TF_x2, wkn_TF_y2, c='none', marker='s', edgecolors='red', s=30)
plt.scatter(pn_TF_x2, pn_TF_y2, c='none', marker='d', edgecolors='blue', s=30)
plt.legend(['Train','Test','Misclassifed by wkNN','Miscalssified by PNN'], loc='lower right')
plt.scatter(train_class[0][: ,0], train_class[0][:,1], color='purple', marker='o',s=10)
plt.scatter(train_class[1][: ,0], train_class[1][:,1], color='yellow', marker='o',s=10)
            
plt.scatter(test_class[0][: ,0], test_class[0][:,1], color='purple', marker='x',s=10)
plt.scatter(test_class[1][: ,0], test_class[1][:,1], color='yellow', marker='x',s=10)


# In[ ]:




