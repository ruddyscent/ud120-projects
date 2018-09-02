#!/usr/bin/python

import sys
import pickle

import pandas as pd
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'total_stock_value',
                 'exercised_stock_options', 'restricted_stock', 'bonus',
                 'deferred_income', 'fraction_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')
    data_df = data_df.replace('NaN', np.nan)
    
### Task 2: Remove outliers
data_df = data_df.drop('TOTAL')
data_df = data_df.drop('LOCKHART EUGENE E')
data_df = data_df.drop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)
data_df['fraction_from_poi'] = data_df['from_poi_to_this_person'] / data_df['to_messages']
data_df['fraction_to_poi'] = data_df['from_this_person_to_poi'] / data_df['from_messages']

### Store to my_dataset for easy export below.
my_dataset = data_df.replace(np.nan, 'NaN').to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.model_selection import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits = 1000, random_state = 42)
for train_idx, test_idx in cv.split(features, labels):
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, features_list)

# C-Support Vector Classification.
from sklearn.svm import SVC

clf = SVC()
clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)

# A decision tree classifier.
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)

# k-nrearest neighbors classifier.
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)

# An AdaBoost classifier.
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)

# A random forest classifier.
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(features_train, labels_train)

test_classifier(clf, my_dataset, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#from tempfile import mkdtemp
#from shutil import rmtree

#cachedir = mkdtemp()
pipe = make_pipeline(MinMaxScaler(), PCA(), GaussianNB())#, memory=cachedir)
tuned_parameters = [{'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8]}]
scores = ['precision', 'recall']

for score in scores:
    print "# Tuning hyper-parameters for %s" % score
    print

    grid = GridSearchCV(pipe, tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    grid.fit(features_train, labels_train)

    print "Best parameters set found on development set:"
    print
    print grid.best_params_
    print  
    print "Grid scores on development set:"
    print  
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params) 
    print  

    print "Detailed classification report:"
    print  
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print  
    labels_true, labels_pred = labels_test, grid.predict(features_test)
    print classification_report(labels_true, labels_pred)
    print  
    
#rmtree(cachedir)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(grid.best_estimator_, my_dataset, features_list)
