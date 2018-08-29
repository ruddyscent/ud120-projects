#!/usr/bin/python3


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = "../tools/python2_lesson14_keys.pkl")
labels, features = targetFeatureSplit(data)



### your code goes here 
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

cls = DecisionTreeClassifier()
cls.fit(features_train, labels_train)
labels_pred = cls.predict(features_test)

print("Predicted POI in test set:", (labels_pred == 1).sum())
print("Number of people in test set:", len(labels_pred))
print("The accuracy of all 0 prediction:", accuracy_score(labels_test, [0]*29))

print("Number of true positive:", sum([p == t == 1 for p, t in zip(labels_pred, labels_test)]))

print("Precision score:", precision_score(labels_test, labels_pred))
print("Recall score:", recall_score(labels_test, labels_pred))
