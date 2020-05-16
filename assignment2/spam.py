#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 entro <entropy1208@yahoo.co.in>
#
# Distributed under terms of the MIT license.

import numpy as np
from sklearn import neighbors, svm
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from stat_tests import friedman_test, nemenyi_multitest


df = pd.read_csv('spambase.data', header=None)
X = df[df.columns[:-1]]
y = np.array(df[df.columns[-1]])
KNeighbors_clf = neighbors.KNeighborsClassifier()
SVM_clf = svm.SVC()
Naive_Bayes_clf = GaussianNB()
classifiers = [KNeighbors_clf, SVM_clf, Naive_Bayes_clf]
measurements = []
for classifier in classifiers:
    scores = cross_validate(classifier, X, y,
                            scoring=['accuracy', 'f1'], cv=10,
                            return_train_score=False)
    measurements.append(scores['test_accuracy'])
    print "Results : ", scores['test_accuracy']
    print "Train time : ", np.average(scores['fit_time'])
    print "F1 measure : ", np.average(scores['test_f1'])
    print "Accuracy : ", np.average(scores['test_accuracy'])
    print "Standard Deviation : ", np.std(scores['test_accuracy'])
friedman_statistic, _, rankings, pivots = friedman_test(*measurements)
ranks = {j: rankings[i] for i, j in enumerate(['KNeighbors',
                                              'SVC', 'Naive_Bayes'])}
critical_value = 7.8
if friedman_statistic > critical_value:
    print '''Null Hypothesis rejected!
The average ranks as a whole display significant differences!
Doing Nemenyi test now!'''
comparisons, z_values, _, _ = nemenyi_multitest(ranks)
print comparisons, z_values
