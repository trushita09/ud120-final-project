#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
				'poi',
				'salary',
				'bonus',
				'total_payments',
				'exercised_stock_options',
				'from_poi_to_this_person',
				'from_this_person_to_poi',
				'shared_receipt_with_poi'
				] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# print data_dict

#plotting scatter plots to see the outliers
from matplotlib import pyplot as plt 

#plotting salary vs bonus
plt.figure()
features_plot = ["salary", "bonus"]
data = featureFormat(data_dict, features_plot)
for point in data:
	salary = point[0]
	bonus = point[1]
	plt.scatter(salary, bonus)
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()

# plotting salary vs total_payments
features_plot = ["salary", "total_payments"]
data = featureFormat(data_dict, features_plot)
for point in data:
	salary = point[0]
	total_pay = point[1]
	plt.scatter(salary, total_pay)
plt.xlabel('Salary')
plt.ylabel('Total Payments')
plt.show()


### Task 2: Remove outliers
data_dict.pop("TOTAL")

plt.figure()
features_plot = ["salary", "bonus"]
data = featureFormat(data_dict, features_plot)
for point in data:
	salary = point[0]
	bonus = point[1]
	plt.scatter(salary, bonus)
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()

### Task 3: Create new feature(s)
#creating fraction of messages sent from poi and fraction of messages to poi
frac_from_poi = []
for i in data_dict:
	if data_dict[i]["from_poi_to_this_person"]=='NaN' or data_dict[i]["to_messages"]=="NaN" :
		frac_from_poi.append(0.)
	else:
		frac_from_poi.append(float(data_dict[i]["from_poi_to_this_person"])/float(data_dict[i]["to_messages"]))

frac_to_poi = []
for i in data_dict:
	if data_dict[i]["from_this_person_to_poi"]=='NaN' or data_dict[i]["from_messages"]=="NaN" :
		frac_to_poi.append(0.)
	else:
		frac_to_poi.append(float(data_dict[i]["from_this_person_to_poi"])/float(data_dict[i]["from_messages"]))

#adding the values to the dataset
cnt = 0
for i in data_dict:
	data_dict[i]['frac_from_poi'] = frac_from_poi[cnt]
	data_dict[i]['frac_to_poi'] = frac_to_poi[cnt]
	cnt += 1

##the new features are not considered in the analysis
features_list.append('frac_from_poi')
features_list.append('frac_to_poi')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)

## Applying feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

## NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)
print "\n"
print "NAIVE BAYES"
print "Accuracy = ", acc
print "Precision = ", prec
print "Recall = ", rec


## SVM CLASSIFIER
from sklearn.svm import SVC
clf = SVC(kernel = "rbf")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)
print "\n"
print "SVM CLASSIFIER"
print "Accuracy = ", acc
print "Precision = ", prec
print "Recall = ", rec


## DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)
print "\n"
print "DECISION TREE"
print "Accuracy = ", acc
print "Precision = ", prec
print "Recall = ", rec
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print '\nFeature Ranking: '
for i in range(7):
   print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])

## ADABOOST CLASSIFIER
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)
print "\n"
print "ADABOOST"
print "Accuracy = ", acc
print "Precision = ", prec
print "Recall = ", rec


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)

#final analysis
#taking Decision Tree Classifer
print("\nAfter Tuning")

#Tuning parameters for decision tree
parameters = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'min_samples_split':[5, 10, 15]}
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "\n"
print "For Decision Tree"
print "Accuracy = ", acc
print clf.best_params_
print classification_report(labels_test, pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)