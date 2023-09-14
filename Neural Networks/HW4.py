import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)
from sklearn import metrics
print("WINE DATASET")
def logistic_regression(x):
    return 1/(1+np.exp(-x))

def loss(WT,X,y):
    nimages = X.shape[1]
    c = WT.shape[0]
    S = np.dot(WT,X)
    P = logistic_regression(S)
    Pyi = P[ y, np.arange(nimages) ]  # select the prob of the true class
    li = -np.log(Pyi)           # cross-entropy
    L = li.sum()    # this is the loss
    # back-prop of the gradient of the loss

    dLdli = np.ones_like(li)

    dLdP = np.zeros_like(P)
    dLdP[ y, np.arange(nimages) ] = dLdli * (-1/Pyi)

    dLdS = np.zeros_like(S)
    for m in range(c):
        dLdS += dLdP[m]*(-P[m]*P)
    dLdS += dLdP*P

    dLdWT = np.dot(dLdS,X.T)  # finally, this is the gradient of the loss

    ypred = np.argmax(P,axis=0)
    return L,dLdWT,ypred

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("hw3_wine.csv",delimiter='\t')
X_df = data.iloc[:,1:]
normalized_df=(X_df-X_df.min())/(X_df.max()-X_df.min())
X = normalized_df.values
Y = data['# class']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, shuffle=True)
Y_train = Y_train.to_numpy()

X_train_T = np.transpose(X_train)
Y_train_T = np.transpose(Y_train)
Y_train_T = Y_train_T-1

X_test_T = np.transpose(X_test)
Y_test_T = np.transpose(Y_test)
Y_test_T = Y_test_T-1
print(X_train_T.shape)

c = len(np.unique(Y))
WT = np.zeros((c,X_train_T.shape[0]))

trainL,train_gradL,train_pred = loss(WT,X_train_T,Y_train_T)
print('Loss for training set: ',trainL)
testL,test_gradL,test_pred = loss(WT,X_test_T,Y_test_T)
print('Loss for testing set: ',testL)

stepsize = 1
number_of_steps = 1000
L_train = np.empty(number_of_steps)
L_test = np.empty(number_of_steps)
f1_score = np.empty(number_of_steps)
train_percent = np.empty(number_of_steps)
test_percent = np.empty(number_of_steps)
for i in range(number_of_steps):
    WT += -stepsize*train_gradL

    L_train[i],train_gradL,train_pred = loss(WT,X_train_T,Y_train_T)
    L_test[i],test_grad,test_pred = loss(WT,X_test_T,Y_test_T)
    train_percent[i] = int(100*(train_pred == Y_train_T).sum()/len(Y_train_T))
    test_percent[i] = int(100*(test_pred == Y_test_T).sum()/len(Y_test_T))
    f1_score[i] = metrics.f1_score(train_pred,Y_train_T, average="macro")

    if(i%100 == 0):
        print('Step-size value: ',stepsize)
        print('Batches value: 10')
        print('Iterations: ',number_of_steps)
        print('Training set loss: ',L_train[i])
        print('Testing set loss: ', L_test[i])
        print('Training set correct percent:', train_percent[i])
        print('Testing correct percent: ', test_percent[i])
        print('F1 Score percent: ', f1_score[i])
        print('----------------------------------------------')

X = np.arange(number_of_steps)
plt.plot(X,L_train)
plt.show()
plt.plot(X,L_test)
plt.show()

print("HOUSE VOTES DATASET")

data = pd.read_csv("hw3_house_votes_84.csv")
X_df = data.iloc[:,:-1]
normalized_df=(X_df-X_df.min())/(X_df.max()-X_df.min())
X = normalized_df.values
Y = data['class']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, shuffle=True)
Y_train = Y_train.to_numpy()

X_train_T = np.transpose(X_train)
Y_train_T = np.transpose(Y_train)

X_test_T = np.transpose(X_test)
Y_test_T = np.transpose(Y_test)

c = len(np.unique(Y))
WT = np.zeros((c,X_train_T.shape[0]))

trainL,train_gradL,train_pred = loss(WT,X_train_T,Y_train_T)
print('Loss for training set: ',trainL)
testL,test_gradL,test_pred = loss(WT,X_test_T,Y_test_T)
print('Loss for testing set: ',testL)

from sklearn import metrics
stepsize = 0.01
number_of_steps = 1000
L_train = np.empty(number_of_steps)
L_test = np.empty(number_of_steps)
f1_score = np.empty(number_of_steps)
train_percent = np.empty(number_of_steps)
test_percent = np.empty(number_of_steps)
for i in range(number_of_steps):
    WT += -stepsize*train_gradL

    L_train[i],train_gradL,train_pred = loss(WT,X_train_T,Y_train_T)
    L_test[i],test_grad,test_pred = loss(WT,X_test_T,Y_test_T)
    train_percent[i] = int(100*(train_pred == Y_train_T).sum()/len(Y_train_T))
    test_percent[i] = int(100*(test_pred == Y_test_T).sum()/len(Y_test_T))
    f1_score[i] = metrics.f1_score(train_pred,Y_train_T, average="macro")

    if(i%100 == 0):
        print('Step-size value: ',stepsize)
        print('Batches value: 10')
        print('Iterations: ',number_of_steps)
        print('Training set loss: ',L_train[i])
        print('Testing set loss: ', L_test[i])
        print('Training set correct percent:', train_percent[i])
        print('Testing correct percent: ', test_percent[i])
        print('F1 Score percent: ', f1_score[i])
        print('----------------------------------------------')

X = np.arange(number_of_steps)
plt.plot(X,L_train)
plt.show()
plt.plot(X,L_test)
plt.show()
