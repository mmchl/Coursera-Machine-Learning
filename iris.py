# -*- coding: utf-8 -*-
"""
Iris
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import tree
from io import StringIO
from IPython.display import Image
import pydotplus

# os.chdir('')

dataset = pd.read_csv('iris.csv')

data_clean = dataset.dropna()
data_clean.dtypes
data_clean.describe()

predictors = data_clean[['SEPAL_LENGTH',
                         'SEPAL_WIDTH',
                         'PETAL_LENGTH',
                         'PETAL_WIDTH']]

targets = data_clean.CLASS

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors,
                                                              targets,
                                                              test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)


out = StringIO()
tree.export_graphviz(classifier, out_file=out)
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
