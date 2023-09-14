import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import statistics
import matplotlib.pyplot as plt
import random
import statistics
from statistics import mode

class Node:
    def __init__(self, attribute_name=None, left=None, middle=None, right=None, information_gain=None, value=None):
        self.attribute_name = attribute_name
        self.left = left
        self.middle = middle
        self.right = right
        self.information_gain = information_gain
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=10):
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
        X = dataset.iloc[:,:-1]
        attributes_list = X.columns
        max_information_gain = 0
        target = dataset.iloc[:,-1:]
        # iterate through all the attributes
        random_attributes_list = random.sample(list(attributes_list),int(len(attributes_list)**(1/2)))
        for attribute in random_attributes_list:
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

data = pd.read_csv("hw3_house_votes_84.csv")
data_target_0 = data.loc[data.iloc[:,-1] == 0]
data_target_1 = data.loc[data.iloc[:,-1] == 1]
splits_0 = []
j = len(data_target_0)//10
for i in range(0,len(data_target_0),j):
    if (len(data_target_0) - i) < (2 * j):
        j = len(data_target_0) - j
    splits_0.append(data_target_0.iloc[i : i+j])
splits_1 = []
j = len(data_target_1)//10
for i in range(0,len(data_target_1),j):
    if (len(data_target_1) - i) < (2 * j):
        j = len(data_target_1) - j
    splits_1.append(data_target_1.iloc[i : i+j])
splits = []
for i in range(10):
    res = pd.concat([splits_0[i],splits_1[i]])
    splits.append(res)



train_sets = []
for i in range(10):
    train_df = pd.concat(splits[:i] + splits[i+1:])
    train_sets.append(train_df)

def bootstrap(train_set,test_set,value):
    bootstrapSetAccuracies = []
    X_test, Y_test = test_set.iloc[:,:-1], test_set.iloc[:,-1:]
    
    Predictions = []
    for i in range(value):
        boot_df = pd.DataFrame(columns = train_sets[0].columns)
        random_elements = list(np.random.randint(low = 0, high = len(train_set), size = len(train_set)))
        boot_df = boot_df.append(train_set.iloc[random_elements])
        X_train, Y_train = boot_df.iloc[:,:-1], boot_df.iloc[:,-1:]
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train,Y_train)
        prediction = dtc.predict(X_test)
        Predictions.append(prediction)
        
    return Predictions


#Cross validation sets -> train_sets
test_sets = splits
values = [1,5,10,20,30,40,50]
value_predictions = {}
for value in values:
    set_predictions = []
    for i in range(len(train_sets)):
        Predictions = bootstrap(train_sets[i],test_sets[i], value)
        final_predictions = []
        for j in range(len(test_sets[i])):
            l = [predict[j][0] for predict in Predictions]
            final_predictions.append(mode(l)) 
        set_predictions.append(final_predictions)
    value_predictions[value] = set_predictions

def get_metrics(y_test,y_pred):
    tp = 0
    fp = 0
    fn = 0 
    for i in range(len(y_test)):
        if y_test[i][0] == y_pred[i] and y_pred[i] == 1:
            tp +=1 
        elif y_pred[i] == 1:
            fp += 1
        elif y_test[i][0] != y_pred[i] and y_pred[i] == 0:
            fn += 1
    return tp/(tp+fp), tp/(tp+fn)

def get_f1_score(precision,recall):
    return 2*(precision*recall)/(precision+recall)

dataset_accuracies = {}
dataset_precision = {}
dataset_recall = {}
dataset_f1_score = {}
for i in range(len(train_sets)):
    accuracies_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    print(f"For Cross-Validation Split {i}")
    for value in values:
        accuracy = accuracy_score(test_sets[i].iloc[:,-1:].values.tolist(),value_predictions[value][i])
        our_precision, our_recall = get_metrics(test_sets[i].iloc[:,-1:].values.tolist(),value_predictions[value][i])
        f1_score = get_f1_score(our_precision,our_recall)
        print(f"Accuracy for value {value} is : {accuracy}")
        print(f"Precision for value {value} is : {our_precision}")
        print(f"Recall for value {value} is : {our_recall}")
        print(f"F1 score fore value {value} is : {f1_score}")
        accuracies_list.append(accuracy)
        precision_list.append(our_precision)
        recall_list.append(our_recall)
        f1_score_list.append(f1_score)
    dataset_accuracies[i] = accuracies_list
    dataset_precision[i] = precision_list
    dataset_recall[i] = recall_list
    dataset_f1_score[i] = f1_score_list

final_accuracies = []
final_precision = []
final_recall = []
final_f1_score = []
for j in range(7):
    total_accuracies_nTree = 0
    mean_accuracies_nTree = 0
    total_precision_nTree = 0
    mean_precision_nTree = 0
    total_recall_nTree = 0
    mean_recall_nTree = 0
    total_f1_score_nTree = 0
    mean_f1score_nTree = 0
    for i in range(10):
        total_accuracies_nTree += dataset_accuracies[i][j]
        total_precision_nTree += dataset_precision[i][j]
        total_recall_nTree += dataset_recall[i][j]
        total_f1_score_nTree += dataset_f1_score[i][j]

    mean_accuracies_nTree = total_accuracies_nTree/10
    mean_precision_nTree = total_precision_nTree/10
    mean_recall_nTree = total_recall_nTree/10
    mean_f1_score_nTree = total_f1_score_nTree/10
    final_accuracies.append(mean_accuracies_nTree)
    final_precision.append(mean_precision_nTree)
    final_recall.append(mean_recall_nTree)
    final_f1_score.append(mean_f1_score_nTree)

print(f"Final Accuracies {final_accuracies}")
print(f"Final Precision {final_precision}")
print(f"Final Recall {final_recall}")
print(f"Final F1 Score {final_f1_score}")

for i in range(7):
    print(f"Random Forest Accuracy nTree={values[i]} = {final_accuracies[i]}")
    print(f"Random Forest Precision nTree={values[i]} = {final_precision[i]}")
    print(f"Random Forest Recall nTree={values[i]} = {final_recall[i]}")
    print(f"Random Forest F1 Score nTree={values[i]} = {final_f1_score[i]}")

from statistics import mean,pstdev

def calculate_deviation(dataset_accuracies, values):
    metrics_per_value = {}
    for j, value in enumerate(values):
        all_accuracies_per_value = []
        for i in range(10):
            all_accuracies_per_value.append(dataset_accuracies[i][j])
        mean_accuracy = mean(all_accuracies_per_value)
        stdev_accuracy = pstdev(all_accuracies_per_value)
        metrics_per_value[value] = [mean_accuracy, stdev_accuracy]

    return metrics_per_value

values = [1,5,10,20,30,40,50]
A = calculate_deviation(dataset_accuracies,values)
B = calculate_deviation(dataset_precision,values)
C = calculate_deviation(dataset_recall,values)
D = calculate_deviation(dataset_f1_score,values)

values,metrics = zip(*A.items())
accuracy,error = zip(*metrics)
plt.errorbar(values, accuracy, yerr = error,fmt = 'o-',color = 'orange', 
       ecolor = 'black', elinewidth = 3, capsize=3)
plt.show()
values,metrics = zip(*B.items())
precision,error = zip(*metrics)
plt.errorbar(values, precision, yerr = error,fmt = 'o-',color = 'green', 
       ecolor = 'black', elinewidth = 3, capsize=3)
plt.show()
values,metrics = zip(*C.items())
recall,error = zip(*metrics)
plt.errorbar(values, recall, yerr = error,fmt = 'o-',color = 'blue', 
       ecolor = 'black', elinewidth = 3, capsize=3)
plt.show()
values,metrics = zip(*D.items())
f1_score,error = zip(*metrics)
plt.errorbar(values, f1_score, yerr = error,fmt = 'o-',color = 'red', 
       ecolor = 'black', elinewidth = 3, capsize=3)
plt.show()
