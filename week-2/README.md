# Coursera Machine Learning (Week 2)
## Dataset information
<a href="https://archive.ics.uci.edu/ml/datasets/Wholesale+customers">https://archive.ics.uci.edu/ml/datasets/Wholesale+customers</a>
<p>The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary units (m.u.) on diverse product categories.</p>
<h5>Attribute information</h5>
<ol>
<li>FRESH: annual spending (m.u.) on fresh products (Continuous)</li>
<li>MILK: annual spending (m.u.) on milk products (Continuous)</li>
<li>GROCERY: annual spending (m.u.) on grocery products (Continuous)</li>
<li>FROZEN: annual spending (m.u.) on frozen products (Continuous)</li>
<li>DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)</li>
<li>DELICATESSEN: annual spending (m.u.) on and delicatessen products (Continuous)</li>
<li>CHANNEL: Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)</li>
<li>REGION: Region - Lisnon, Oporto or Other (Nominal)</li>
</ol>
<h5>Descriptive Statistics</h5>
<pre><code>
                 (Minimum, Maximum, Mean, Std. Deviation)
FRESH            (3, 112151, 12000.30, 12647.329)
MILK	           (55, 73498, 5796.27, 7380.377)
GROCERY	         (3, 92780, 7951.28, 9503.163)
FROZEN	         (25, 60869, 3071.93, 4854.673)
DETERGENTS_PAPER (3, 40827, 2881.49, 4767.854)
DELICATESSEN     (3, 47943, 1524.87, 2820.106)

REGION	      Frequency
Lisbon	      77 
Oporto	      47 
Other Region	316 
Total	        440

CHANNEL	Frequency 
Horeca	298 
Retail	142 
Total	  440 
</code></pre>
## Results
<pre><code>
In [2]: data_clean.dtypes
Out[2]: 
Channel              int64
Region              object
Fresh                int64
Milk                 int64
Grocery              int64
Frozen               int64
Detergents_Paper     int64
Delicassen           int64
dtype: object

In [3]: data_clean.describe()
Out[3]: 
          Channel          Fresh          Milk       Grocery        Frozen  \
count  440.000000     440.000000    440.000000    440.000000    440.000000   
mean     1.322727   12000.297727   5796.265909   7951.277273   3071.931818   
std      0.468052   12647.328865   7380.377175   9503.162829   4854.673333   
min      1.000000       3.000000     55.000000      3.000000     25.000000   
25%      1.000000    3127.750000   1533.000000   2153.000000    742.250000   
50%      1.000000    8504.000000   3627.000000   4755.500000   1526.000000   
75%      2.000000   16933.750000   7190.250000  10655.750000   3554.250000   
max      2.000000  112151.000000  73498.000000  92780.000000  60869.000000   

       Detergents_Paper    Delicassen  
count        440.000000    440.000000  
mean        2881.493182   1524.870455  
std         4767.854448   2820.105937  
min            3.000000      3.000000  
25%          256.750000    408.250000  
50%          816.500000    965.500000  
75%         3922.000000   1820.250000  
max        40827.000000  47943.000000  

In [4]: data_clean.shape
Out[4]: (440, 8)

In [5]: print(pred_train.shape, tar_train.shape)
(264, 7) (264,)

In [6]: print(pred_test.shape, tar_test.shape)
(176, 7) (176,)

In [7]: classifier = RandomForestClassifier(n_estimators=25)

In [8]: classifier = classifier.fit(pred_train, tar_train)

In [9]: predictions = classifier.predict(pred_test)

In [10]: sklearn.metrics.confusion_matrix(tar_test,predictions)
Out[10]: 
array([[  1,   0,  30],
       [  1,   0,  14],
       [  9,   2, 119]])

In [11]: sklearn.metrics.accuracy_score(tar_test, predictions)
Out[11]: 0.68181818181818177

In [12]: model = ExtraTreesClassifier()

In [13]: model.fit(pred_train,tar_train)
Out[13]: 
ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

In [14]: print(model.feature_importances_)
[ 0.01914095  0.13674427  0.17661013  0.16598989  0.18082758  0.16871081
  0.15197637]

In [15]: trees=range(25)
    ...: accuracy=np.zeros(25)
    ...: 
    ...: for idx in range(len(trees)):
    ...:    classifier=RandomForestClassifier(n_estimators=idx + 1)
    ...:    classifier=classifier.fit(pred_train,tar_train)
    ...:    predictions=classifier.predict(pred_test)
    ...:    accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
    ...:    

In [16]: plt.cla()
    ...: plt.plot(trees, accuracy)
    ...: 
Out[16]: [<matplotlib.lines.Line2D at 0x1a1b354748>]
</code></pre>
<img src="https://github.com/mmchl/Coursera-Machine-Learning/blob/master/week-2/wholesale.png">
<h2>Conclusion</h2>
<p>There is no significant differences in algorithm accuracies after 13 trees.</p>
