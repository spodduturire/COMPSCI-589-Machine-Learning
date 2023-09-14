import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import statistics
import matplotlib.pyplot as plt

class Node:
    def __init__(self, attribute_name=None, left=None, middle=None, right=None, information_gain=None, value=None):
        self.attribute_name = attribute_name
        self.left = left
        self.middle = middle
        self.right = right
        self.information_gain = information_gain
        
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2):
        self.root=None
        self.min_samples_split = min_samples_split
    
    def build_tree(self, dataset):
        X, Y = dataset.iloc[:,:-1], dataset.iloc[:,-1:]
        #num_samples = len(X)
        
        if len(X)>=self.min_samples_split:
            best_split= self.get_best_split(dataset)
            if best_split["information_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"])
                middle_subtree = self.build_tree(best_split["dataset_middle"])
                right_subtree = self.build_tree(best_split["dataset_right"])
                return Node(best_split["attribute_name"], left_subtree, 
                            middle_subtree, right_subtree, best_split["information_gain"])
                
        leaf_value = self.calculate_leaf_target(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset):
        best_split = {}
        best_split["information_gain"] = -1
        attributes_list = dataset.columns
        max_information_gain = 0
        target = dataset.iloc[:,-1:]
        # iterate through all the attributes
        for attribute in attributes_list:
            # ignore the attributes upon which we already split once 
            if len(dataset[attribute].unique()) > 1: 
                # three-way splitting here
                left, middle, right = dataset.loc[dataset[attribute]==0],dataset.loc[dataset[attribute]==1],dataset.loc[dataset[attribute]==2]
                if len(left)>0 and len(middle)>0 and len(right)>0:
                    # calculate information gain
                    gain = self.get_information_gain(target,left.iloc[:,-1:],middle.iloc[:,-1:],right.iloc[:,-1:])
                    # checks for attribute with highest information gain
                    if gain > max_information_gain:
                        best_split["attribute_name"] = attribute
                        best_split["dataset_left"] = left
                        best_split["dataset_middle"] = middle
                        best_split["dataset_right"] = right
                        best_split["information_gain"] = gain
                        max_information_gain = gain
        return best_split

    def get_entropy(self, target):
        entropy = 0
        total_count = target.value_counts()
        for vote in total_count.keys():
            entropy -= (total_count[vote]/len(target))*np.log2(total_count[vote]/len(target))
        return entropy

    def get_information_gain(self, parent, left, middle, right):
        gain = 0.0
        gain = self.get_entropy(parent) - ((len(left)/len(parent))*self.get_entropy(left) + (len(middle)/len(parent))*self.get_entropy(middle) + (len(right)/len(parent))*self.get_entropy(right))
        return gain

    def calculate_leaf_target(self, target):
        Y = target.values.tolist()
        return max(Y, key=Y.count)
            
    def fit(self, X, Y):
        dataset = pd.concat([X, Y], axis=1)
        self.root = self.build_tree(dataset)
        
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x = X.iloc[i]
            predictions.append(self.make_prediction(x,self.root))
        return predictions
    
    def make_prediction(self, x, tree):

        if tree.value!=None: return tree.value
        attribute_value = x[tree.attribute_name]
        if attribute_value==0:
            return self.make_prediction(x, tree.left)
        elif attribute_value==1:
            return self.make_prediction(x, tree.middle)
        else:
            return self.make_prediction(x, tree.right)

data = pd.read_csv("house_votes_84.csv")
X, Y = data.iloc[:,:-1], data.iloc[:,-1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, shuffle=True)
accuracies_train = []
accuracies_test = []

for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train,Y_train)
    accuracy = accuracy_score(Y_train, dtc.predict(X_train))
    accuracies_train.append(accuracy)
print(accuracies_train)

for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train,Y_train)
    accuracy = accuracy_score(Y_test, dtc.predict(X_test))
    accuracies_test.append(accuracy)
print(accuracies_test)

print(f"Mean Train -> {statistics.mean(accuracies_train)}")
print(f"Standard Deviation Train -> {statistics.pstdev(accuracies_train)}")
print(f"Mean Test -> {statistics.mean(accuracies_test)}")
print(f"Standard Deviation Test -> {statistics.pstdev(accuracies_test)}")
print(Counter(accuracies_train))
print(Counter(accuracies_test))

fig, (ax1, ax2) = plt.subplots(2, sharey=True)
fig.suptitle('Train v/s Test')
ax1.hist(accuracies_train)
ax2.hist(accuracies_test)
plt.show()