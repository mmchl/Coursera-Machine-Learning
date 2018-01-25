# Coursera-Machine-Learning (Week 1)
## Dataset information
<a href="https://archive.ics.uci.edu/ml/datasets/iris">https://archive.ics.uci.edu/ml/datasets/iris</a>
<p>Attributes</p>
<ul>
<li>Sepal length in cm</li>
<li>Sepal width in cm</li>
<li>Petal length in cm</li>
<li>Petal width in cm</li>
<li>Class:</li>
<ul><li>Iris Setosa</li>
<li>Iris Versicolour</li>
<li>Iris Virginica</li>
  </ul>
</ul>

## Results
<pre><code>
In [3]: data_clean.dtypes
Out[3]: 
SEPAL_LENGTH    float64
SEPAL_WIDTH     float64
PETAL_LENGTH    float64
PETAL_WIDTH     float64
CLASS            object
dtype: object

In [4]: data_clean.describe()
Out[4]: 
       SEPAL_LENGTH  SEPAL_WIDTH  PETAL_LENGTH  PETAL_WIDTH
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000

In [5]: pred_train.shape
Out[5]: (90, 4)

In [6]: pred_test.shape
Out[6]: (60, 4)

In [7]: tar_train.shape
Out[7]: (90,)

In [8]: tar_test.shape
Out[8]: (60,)

In [9]: classifier=DecisionTreeClassifier()
   ...: classifier=classifier.fit(pred_train,tar_train)
   ...: 

In [10]: predictions=classifier.predict(pred_test)

In [11]: sklearn.metrics.confusion_matrix(tar_test,predictions)
Out[11]: 
array([[23,  0,  0],
       [ 0, 17,  2],
       [ 0,  0, 18]])

In [12]: sklearn.metrics.accuracy_score(tar_test, predictions)
Out[12]: 0.96666666666666667

In [13]: out = StringIO()
    ...: tree.export_graphviz(classifier, out_file=out)
    ...: graph=pydotplus.graph_from_dot_data(out.getvalue())
    ...: Image(graph.create_png())
    ...: 
Out[13]: 
</code></pre>
<img src="https://github.com/mmchl/Coursera-Machine-Learning/blob/master/week-1/iris.png">
