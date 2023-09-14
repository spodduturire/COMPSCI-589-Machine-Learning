#Required library imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import statistics
import matplotlib.pyplot as plt

#Function to calculate the euclidean distance
def eucledian_distance(row1,row2):
    total = 0.0
    
    for i in range(len(row1)):
        dist=(row1[i] - row2[i])**2
        total = total + dist
    return math.sqrt(total)

#Function to get neighbours
def get_neighbours(trainset,testrow,k):
    distance_list = []
    closest_neighbors = []
    for i in range(len(trainset)):
        distance = eucledian_distance(trainset[i],testrow)
        distance_list.append((i,distance))
    res = sorted(distance_list, key = lambda x : x[1])[:k]
    closest_neighbors = [r[0] for r in res]        
    return closest_neighbors


def get_prediction(trainset,testrow,k):
    closest_neighbors = get_neighbours(trainset,testrow,k)
    target_list = []
    for neighbor in closest_neighbors:
        target_list.append(Y_train[neighbor])
    prediction = max(target_list,key = target_list.count)
    return prediction

def apply_knn(trainset,testset,k):
    prediction_list = []
    for testrow in testset:
        prediction_list.append(get_prediction(trainset,testrow,k))
    
    return prediction_list


# Import data into dataframe
df=pd.read_csv('iris.csv',header=None)
#Separate the data from the target values and normalize the data
X_df = df[[0,1,2,3]]
normalized_df=(X_df-X_df.min())/(X_df.max()-X_df.min())
X = normalized_df.values
Y = df[[4]].values

k_accuracies_train ={}
k_accuracies_test = {}
for i in range(20):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, shuffle=True)
    for k in range(1,53,2):
        train_prediction = apply_knn(X_train,X_train,k)
        train_accuracy = metrics.accuracy_score(Y_train,train_prediction)
        if k not in k_accuracies_train:
            k_accuracies_train[k] = [train_accuracy]
        else:
            k_accuracies_train[k].append(train_accuracy)
            
    for k in range(1,53,2):
        test_prediction = apply_knn(X_train,X_test,k)
        test_accuracy = metrics.accuracy_score(Y_test,test_prediction)
        if k not in k_accuracies_test:
            k_accuracies_test[k] = [test_accuracy]
        else:
            k_accuracies_test[k].append(test_accuracy)

print(f"Training Accuracies -> {k_accuracies_train}")
print(f"Testing Accuracies -> {k_accuracies_test}")

for k in k_accuracies_train:
    k_accuracies_train[k] = (statistics.mean(k_accuracies_train[k]),statistics.pstdev(k_accuracies_train[k]))


# k_accuracies_test
for k in k_accuracies_test:
    k_accuracies_test[k] = (statistics.mean(k_accuracies_test[k]),statistics.pstdev(k_accuracies_test[k]))

#Plotting the accuracies

x_train,y_train = zip(*k_accuracies_train.items())
y_train_acc,y_train_err = zip(*y_train)

x_test,y_test = zip(*k_accuracies_test.items())
y_test_acc,y_test_err = zip(*y_test)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

fig.suptitle('Train v/s Test')
fig.text(0.5, 0.04, 'K-values', ha='center', va='center')
fig.text(0.04, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical')
ax1.errorbar(x_train, y_train_acc, yerr = y_train_err,fmt = 'o-',color = 'orange', 
       ecolor = 'black', elinewidth = 3, capsize=3)

ax2.errorbar(x_test, y_test_acc, yerr = y_test_err,fmt = 'o-',color = 'green', 
       ecolor = 'black', elinewidth = 3, capsize=3)


plt.show()
