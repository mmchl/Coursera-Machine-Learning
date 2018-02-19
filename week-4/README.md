<h1>Coursera Machine Learing (Week 4)</h1>
<h2>Dataset information</h2>
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
<h2>Results</h2>
<pre><code>
In [10]: from pandas import Series, DataFrame
    ...: import pandas as pd
    ...: import numpy as np
    ...: import matplotlib.pylab as plt
    ...: from sklearn.cross_validation import train_test_split
    ...: from sklearn import preprocessing
    ...: from sklearn.cluster import KMeans
    ...: 
    ...: # Loading the dataset
    ...: data = pd.read_csv('wholesale.csv')
    ...: 

In [11]: # Loading the dataset
    ...: data = pd.read_csv('wholesale.csv')
    ...: # Mapping Columns
    ...: data.columns = map(str.upper, data.columns)
    ...: 
    ...: data_clean = data.dropna()
    ...: 
    ...: cluster=data_clean
    ...: cluster.describe()
    ...: 
Out[11]: 
          CHANNEL          FRESH          MILK       GROCERY        FROZEN  \
count  440.000000     440.000000    440.000000    440.000000    440.000000   
mean     1.322727   12000.297727   5796.265909   7951.277273   3071.931818   
std      0.468052   12647.328865   7380.377175   9503.162829   4854.673333   
min      1.000000       3.000000     55.000000      3.000000     25.000000   
25%      1.000000    3127.750000   1533.000000   2153.000000    742.250000   
50%      1.000000    8504.000000   3627.000000   4755.500000   1526.000000   
75%      2.000000   16933.750000   7190.250000  10655.750000   3554.250000   
max      2.000000  112151.000000  73498.000000  92780.000000  60869.000000   

       DETERGENTS_PAPER    DELICASSEN  
count        440.000000    440.000000  
mean        2881.493182   1524.870455  
std         4767.854448   2820.105937  
min            3.000000      3.000000  
25%          256.750000    408.250000  
50%          816.500000    965.500000  
75%         3922.000000   1820.250000  
max        40827.000000  47943.000000  

In [12]: # Dict
    ...: regions_replacer = {'Other': 3, 'Lisbon': 1, 'Oporto': 2}
    ...: data_clean.REGION = data_clean.REGION.replace(regions_replacer)
    ...: 
    ...: data_clean.head()
    ...: 
Out[12]: 
   CHANNEL  REGION  FRESH  MILK  GROCERY  FROZEN  DETERGENTS_PAPER  DELICASSEN
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185

In [13]: clustervar = cluster.copy()
    ...: # Cycle for converting rows to float64
    ...: for col in clustervar.columns:
    ...:     clustervar[col] = preprocessing.scale(clustervar[col].astype('float64'))
    ...:     

In [16]: clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)
    ...: 
    ...: from scipy.spatial.distance import cdist
    ...: clusters=range(1,20)
    ...: meandist=[]
    ...: 
    ...: for k in clusters:
    ...:     model=KMeans(n_clusters=k)
    ...:     model.fit(clus_train)
    ...:     clusassign=model.predict(clus_train)
    ...:     meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
    ...:     / clus_train.shape[0])
    ...:     

In [17]: plt.plot(clusters, meandist)
    ...: plt.xlabel('Number of clusters')
    ...: plt.ylabel('Average distance')
    ...: plt.title('Selecting k with the Elbow Method')
    ...: # Show the plot
    ...: plt.show()
    ...: 
</code></pre>
<img src="https://github.com/mmchl/Coursera-Machine-Learning/blob/master/week-4/selecting-k.png">
<pre><code>
In [18]: # Interpret 3 cluster solution
    ...: model3=KMeans(n_clusters=3)
    ...: model3.fit(clus_train)
    ...: clusassign=model3.predict(clus_train)
    ...: 

In [19]: # plot clusters
    ...: from sklearn.decomposition import PCA
    ...: pca_2 = PCA(2)
    ...: plot_columns = pca_2.fit_transform(clus_train)
    ...: plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
    ...: plt.xlabel('Canonical variable 1')
    ...: plt.ylabel('Canonical variable 2')
    ...: plt.title('Scatterplot of Canonical Variables for 3 Clusters')
    ...: plt.show()
    ...: 
</code></pre>
<img src="https://github.com/mmchl/Coursera-Machine-Learning/blob/master/week-4/plot.png">
<pre><code>
In [20]: # create a unique identifier variable from the index for the
    ...: # cluster training data to merge with the cluster assignment variable
    ...: clus_train.reset_index(level=0, inplace=True)
    ...: # create a list that has the new index variable
    ...: cluslist=list(clus_train['index'])
    ...: # create a list of cluster assignments
    ...: labels=list(model3.labels_)
    ...: # combine index variable list with cluster assignment list into a dictionary
    ...: newlist=dict(zip(cluslist, labels))
    ...: print(newlist)
    ...: 
{138: 0, 201: 1, 378: 0, 283: 0, 171: 1, 335: 1, 203: 0, 184: 0, 379: 1, 185: 0, 38: 1, 223: 1, 148: 0, 182: 0, 12: 1, 30: 0, 144: 0, 373: 1, 191: 0, 285: 0, 355: 0, 309: 1, 372: 0, 105: 0, 360: 0, 288: 0, 245: 1, 19: 0, 29: 0, 78: 0, 394: 0, 218: 1, 344: 0, 408: 1, 71: 0, 94: 1, 209: 1, 426: 0, 35: 1, 128: 0, 74: 1, 53: 1, 22: 0, 173: 1, 240: 0, 364: 0, 221: 0, 310: 0, 163: 1, 295: 1, 90: 0, 347: 1, 388: 0, 436: 0, 367: 0, 234: 0, 422: 0, 216: 1, 4: 1, 199: 0, 406: 0, 258: 0, 177: 0, 143: 0, 425: 0, 152: 0, 80: 0, 117: 0, 156: 1, 277: 0, 249: 0, 418: 1, 104: 0, 61: 2, 108: 1, 127: 1, 81: 1, 433: 0, 274: 0, 44: 1, 164: 1, 284: 0, 89: 0, 399: 0, 287: 0, 413: 0, 390: 0, 263: 0, 328: 0, 166: 1, 220: 0, 210: 0, 167: 0, 428: 0, 232: 0, 354: 0, 7: 1, 306: 1, 115: 0, 393: 0, 368: 0, 362: 0, 421: 1, 136: 0, 318: 0, 273: 0, 352: 0, 375: 0, 145: 1, 321: 0, 85: 2, 419: 0, 49: 1, 37: 1, 34: 0, 363: 0, 219: 0, 23: 2, 252: 0, 404: 0, 291: 0, 228: 0, 10: 1, 66: 0, 28: 1, 297: 1, 207: 1, 124: 0, 69: 0, 161: 0, 194: 0, 137: 0, 319: 1, 112: 0, 100: 1, 293: 1, 332: 0, 345: 0, 110: 0, 187: 0, 122: 0, 281: 1, 92: 1, 349: 1, 195: 0, 62: 1, 423: 1, 174: 0, 397: 0, 45: 1, 381: 0, 130: 0, 50: 0, 97: 0, 197: 1, 215: 0, 238: 0, 384: 0, 434: 0, 346: 1, 63: 1, 316: 0, 169: 0, 247: 0, 257: 0, 251: 1, 269: 0, 323: 0, 431: 0, 153: 0, 391: 0, 40: 0, 338: 0, 343: 1, 65: 1, 206: 0, 8: 0, 77: 1, 58: 0, 250: 0, 119: 0, 141: 0, 88: 0, 403: 0, 307: 0, 300: 1, 254: 0, 60: 1, 256: 0, 301: 1, 296: 0, 315: 1, 400: 0, 46: 1, 86: 2, 109: 1, 133: 0, 16: 1, 303: 1, 193: 1, 401: 0, 116: 0, 330: 0, 294: 0, 27: 0, 140: 0, 198: 0, 14: 1, 87: 0, 265: 0, 262: 0, 222: 0, 243: 0, 259: 0, 351: 1, 103: 0, 149: 0, 139: 0, 67: 1, 3: 0, 342: 0, 56: 1, 304: 1, 233: 0, 51: 0, 1: 1, 18: 1, 268: 1, 412: 0, 312: 1, 25: 1, 183: 2, 357: 1, 186: 0, 43: 1, 129: 0, 359: 0, 244: 0, 255: 0, 270: 0, 414: 0, 154: 0, 371: 0, 439: 0, 70: 0, 398: 0, 158: 1, 416: 1, 271: 0, 146: 0, 205: 1, 278: 0, 314: 0, 331: 1, 64: 0, 325: 0, 432: 0, 387: 0, 168: 0, 76: 0, 290: 0, 350: 0, 118: 0, 213: 0, 353: 1, 358: 0, 180: 0, 348: 0, 380: 0, 99: 0, 361: 0, 135: 0, 176: 0, 424: 1, 84: 1, 437: 1, 39: 0, 340: 1, 2: 1, 311: 0, 305: 1, 68: 0, 208: 0, 427: 0, 253: 0, 334: 0, 339: 0, 409: 0, 111: 1, 224: 0, 32: 0, 73: 1, 47: 2, 126: 0, 113: 0, 96: 1, 225: 0, 214: 1, 57: 1, 123: 1, 106: 1, 83: 0, 17: 0, 230: 1, 98: 0, 322: 0, 382: 0, 365: 1}

In [21]: # convert newlist dictionary to a dataframe
    ...: newclus=DataFrame.from_dict(newlist, orient='index')
    ...: print(newclus)
    ...: 
     0
138  0
201  1
378  0
283  0
171  1
335  1
203  0
184  0
379  1
185  0
38   1
223  1
148  0
182  0
12   1
30   0
144  0
373  1
191  0
285  0
355  0
309  1
372  0
105  0
360  0
288  0
245  1
19   0
29   0
78   0
..  ..
2    1
311  0
305  1
68   0
208  0
427  0
253  0
334  0
339  0
409  0
111  1
224  0
32   0
73   1
47   2
126  0
113  0
96   1
225  0
214  1
57   1
123  1
106  1
83   0
17   0
230  1
98   0
322  0
382  0
365  1

[308 rows x 1 columns]

In [22]: # rename the cluster assignment column
    ...: newclus.columns = ['cluster']

In [23]: # now do the same for the cluster assignment variable
    ...: # create a unique identifier variable from the index for the
    ...: # cluster assignment dataframe
    ...: # to merge with cluster training data
    ...: newclus.reset_index(level=0, inplace=True)
    ...: # merge the cluster assignment dataframe with the cluster training variable dataframe
    ...: # by the index variable
    ...: merged_train=pd.merge(clus_train, newclus, on='index')
    ...: merged_train.head(n=100)
    ...: # cluster frequencies
    ...: merged_train.cluster.value_counts()
    ...: 
Out[23]: 
0    204
1     98
2      6
Name: cluster, dtype: int64

In [24]: # FINALLY calculate clustering variable means by cluster
    ...: clustergrp = merged_train.groupby('cluster').mean()
    ...: print ("Clustering variable means by cluster")
    ...: print(clustergrp)
    ...: 
Clustering variable means by cluster
              index   CHANNEL    REGION     FRESH      MILK   GROCERY  \
cluster                                                                 
0        237.970588 -0.679812 -0.017805  0.096795 -0.339362 -0.418092   
1        189.193878  1.448652  0.049717 -0.359304  0.536792  0.782276   
2         80.833333  1.092160  0.590668  1.460047  5.831048  4.118126   

           FROZEN  DETERGENTS_PAPER  DELICASSEN  
cluster                                          
0        0.155980         -0.426792   -0.081564  
1       -0.328909          0.840448    0.028019  
2        1.247826          3.466577    4.002378  

In [25]: region_data=data_clean['REGION']
    ...: class_train, gpa_test = train_test_split(region_data, test_size=.3, random_state=123)
    ...: class_train1 = pd.DataFrame(class_train)
    ...: class_train1.reset_index(level=0, inplace=True)
    ...: merged_train_all = pd.merge(class_train1, merged_train, on='index')
    ...: print(merged_train_all)
    ...: 
     index  REGION_x   CHANNEL  REGION_y     FRESH      MILK   GROCERY  \
0      138         3 -0.690297  0.590668  0.121642 -0.208799 -0.307329   
1      201         1  1.448652 -1.995342 -0.594976  1.166949  1.765286   
2      378         3 -0.690297  0.590668 -0.694636 -0.339429 -0.637279   
3      283         3 -0.690297  0.590668  1.089746 -0.296157 -0.625585   
4      171         3  1.448652  0.590668 -0.934089  2.721890  1.249924   
5      335         2  1.448652 -0.702337  1.193839  0.138461  0.299053   
6      203         1 -0.690297 -1.995342 -0.903771 -0.693336 -0.604199   
7      184         3 -0.690297  0.590668 -0.924036 -0.661730 -0.341462   
8      379         3  1.448652  0.590668 -0.629489 -0.085766  0.257020   
9      185         3 -0.690297  0.590668 -0.303199  0.088407 -0.717659   
10      38         3  1.448652  0.590668 -0.586506  1.347362  0.922608   
11     223         1  1.448652 -1.995342 -0.729070 -0.443472 -0.282994   
12     148         3 -0.690297  0.590668 -0.480988 -0.707037 -0.780552   
13     182         3 -0.690297  0.590668 -0.894985  0.371234  0.270399   
14      12         3  1.448652  0.590668  1.560499  0.884800  0.400925   
15      30         3 -0.690297  0.590668  0.539439 -0.296564  0.332449   
16     144         3 -0.690297  0.590668  0.540389 -0.287476 -0.628219   
17     373         3  1.448652  0.590668  0.243467  0.062498 -0.058287   
18     191         3 -0.690297  0.590668  0.128529 -0.746239 -0.692165   
19     285         3 -0.690297  0.590668  2.236509 -0.699441 -0.458398   
20     355         3 -0.690297  0.590668 -0.934881 -0.687639 -0.625690   
21     309         2  1.448652 -0.702337 -0.877253  2.015567  0.591605   
22     372         3 -0.690297  0.590668 -0.642471 -0.270655 -0.664774   
23     105         3 -0.690297  0.590668  0.283916 -0.648708 -0.419419   
24     360         3 -0.690297  0.590668  0.613135 -0.396673 -0.626323   
25     288         3 -0.690297  0.590668  0.337190 -0.705680 -0.701120   
26     245         1  1.448652 -1.995342 -0.707539  0.048526  0.628371   
27      19         3 -0.690297  0.590668 -0.334071 -0.447812  0.159362   
28      29         3 -0.690297  0.590668  2.460843 -0.501394 -0.562798   
29      78         3 -0.690297  0.590668 -0.097705 -0.626869 -0.619896   
..     ...       ...       ...       ...       ...       ...       ...   
278      2         3  1.448652  0.590668 -0.447029  0.408538 -0.028157   
279    311         2 -0.690297 -0.702337  1.395929 -0.469516  0.034630   
280    305         2  1.448652 -0.702337 -0.930685  0.968902  0.094889   
281     68         3 -0.690297  0.590668 -0.756300  0.198554 -0.416996   
282    208         1 -0.690297 -1.995342 -0.828255 -0.277845 -0.222629   
283    427         3 -0.690297  0.590668  1.504930  1.477314 -0.265717   
284    253         1 -0.690297 -1.995342  1.387301  0.293643  0.949682   
285    334         2  1.448652 -0.702337  0.381756 -0.660374 -0.548681   
286    339         2 -0.690297 -0.702337 -0.742764 -0.625105 -0.275935   
287    409         3 -0.690297  0.590668 -0.260612 -0.293308 -0.195028   
288    111         3  1.448652  0.590668  0.045809  0.721344  1.013207   
289    224         1 -0.690297 -1.995342 -0.374125 -0.696863 -0.679734   
290     32         3 -0.690297  0.590668  0.762427 -0.607471 -0.533616   
291     73         3  1.448652  0.590668  0.625246 -0.062977  0.080246   
292     47         3  1.448652  0.590668  2.569923  6.573905  5.016638   
293    126         3 -0.690297  0.590668  0.571419 -0.536662 -0.662984   
294    113         3 -0.690297  0.590668  0.192964 -0.474942 -0.655082   
295     96         3  1.448652  0.590668 -0.948100 -0.431399  0.017564   
296    225         1 -0.690297 -1.995342  0.053804 -0.346347 -0.399719   
297    214         1  1.448652 -1.995342 -0.762949  0.102379  0.359523   
298     57         3  1.448652  0.590668 -0.521121  0.561142  0.267133   
299    123         3  1.448652  0.590668 -0.065725  0.674545  0.090886   
300    106         3  1.448652  0.590668 -0.834825  0.073350  0.289994   
301     83         3 -0.690297  0.590668  0.702425 -0.574237 -0.649815   
302     17         3 -0.690297  0.590668 -0.484788  0.048933 -0.528665   
303    230         1  1.448652 -1.995342 -0.073482  0.026144 -0.246122   
304     98         3 -0.690297  0.590668 -0.910104 -0.771063 -0.755690   
305    322         2 -0.690297 -0.702337  0.307189 -0.689538 -0.488422   
306    382         3 -0.690297  0.590668  1.777392  0.222292  0.054541   
307    365         3  1.448652  0.590668 -0.578511  0.044999  0.007767   

       FROZEN  DETERGENTS_PAPER  DELICASSEN  cluster  
0   -0.601534         -0.552762    0.619876        0  
1    0.098382          2.383972    0.055426        1  
2   -0.575344         -0.590768   -0.145859        0  
3    1.491202         -0.539114   -0.050009        0  
4   -0.499248          1.237079    1.677422        1  
5   -0.352006          0.258168    0.218016        1  
6   -0.536781         -0.404729   -0.534939        0  
7   -0.618238         -0.534914   -0.537424        0  
8   -0.606690         -0.434335   -0.477784        1  
9   -0.183729         -0.592868   -0.453644        0  
10  -0.626693          0.855551   -0.387614        1  
11   0.523817         -0.439585   -0.058529        1  
12   0.202111         -0.545623   -0.387259        0  
13  -0.542142          0.845263   -0.485949        0  
14  -0.574313          0.209873    0.499176        1  
15  -0.396756         -0.156956    0.510536        0  
16  -0.609164         -0.496698   -0.469974        0  
17  -0.323341         -0.202731    0.563786        1  
18  -0.449961         -0.596017   -0.511509        0  
19  -0.418616         -0.513497   -0.534939        0  
20  -0.582974         -0.566411   -0.496244        0  
21  -0.331384          0.832454   -0.255199        1  
22  -0.452848         -0.521686    1.172967        0  
23  -0.445837         -0.575860   -0.052139        0  
24  -0.097116         -0.506777   -0.362764        0  
25  -0.458623         -0.511607   -0.449739        0  
26  -0.586068          1.270675    0.446991        1  
27  -0.495536         -0.076325   -0.363474        0  
28  -0.386033         -0.372602   -0.249164        0  
29  -0.201258         -0.541843   -0.482044        0  
..        ...               ...         ...      ...  
278 -0.137536          0.133232    2.243293        1  
279 -0.005348         -0.527145   -0.499794        0  
280 -0.468728          0.215752   -0.466424        1  
281  0.577022         -0.439585    0.557396        0  
282 -0.250133          0.104885   -0.255199        0  
283  2.476735         -0.512867   -0.128464        0  
284 -0.544411         -0.528825   -0.047524        0  
285  1.750216         -0.535334    0.696201        0  
286  1.342929         -0.484730    0.148081        0  
287 -0.149084         -0.159266    1.282307        0  
288 -0.467490          0.750773   -0.002084        1  
289 -0.471821         -0.590348   -0.307384        0  
290 -0.578644         -0.412288   -0.397554        0  
291  1.043495         -0.444624   -0.311289        1  
292  0.971318          4.470300    1.753747        2  
293  1.056487         -0.531765   -0.369864        0  
294  0.030535         -0.482210    0.012826        0  
295 -0.603596          0.208403   -0.464294        1  
296 -0.497392         -0.445254   -0.262299        0  
297 -0.445218          0.645785   -0.260524        1  
298 -0.625662          0.984897   -0.086219        1  
299 -0.181048         -0.190132   -0.490564        1  
300 -0.606071          0.829094    0.108676        1  
301 -0.155065         -0.489559   -0.267269        0  
302 -0.460479         -0.527355    1.048362        0  
303  1.082471         -0.404519    0.217306        1  
304 -0.448930         -0.593288   -0.494469        0  
305  0.130140         -0.296381   -0.460034        0  
306 -0.109696         -0.245776    0.485686        0  
307 -0.299626          0.342998    0.560946        1  

[308 rows x 11 columns]

In [26]: sub1 = merged_train_all[['REGION_x', 'cluster']].dropna()
    ...: 
    ...: import statsmodels.formula.api as smf
    ...: import statsmodels.stats.multicomp as multi
    ...: 
    ...: gpamod = smf.ols(formula='REGION_x ~ C(cluster)', data=sub1).fit()
    ...: print (gpamod.summary())
    ...: 
    ...: print ('means for REGION by cluster')
    ...: m1= sub1.groupby('cluster').mean()
    ...: print (m1)
    ...: 
    ...: print ('standard deviations for REGION by cluster')
    ...: m2= sub1.groupby('cluster').std()
    ...: print (m2)
    ...: 
    ...: mc1 = multi.MultiComparison(sub1['REGION_x'], sub1['cluster'])
    ...: res1 = mc1.tukeyhsd()
    ...: print(res1.summary())
    ...: 
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               REGION_x   R-squared:                       0.008
Model:                            OLS   Adj. R-squared:                  0.001
Method:                 Least Squares   F-statistic:                     1.187
Date:                Tue, 20 Feb 2018   Prob (F-statistic):              0.306
Time:                        01:11:59   Log-Likelihood:                -353.17
No. Observations:                 308   AIC:                             712.3
Df Residuals:                     305   BIC:                             723.5
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
Intercept           2.5294      0.054     47.202      0.000       2.424       2.635
C(cluster)[T.1]     0.0522      0.094      0.555      0.579      -0.133       0.237
C(cluster)[T.2]     0.4706      0.317      1.484      0.139      -0.153       1.094
==============================================================================
Omnibus:                       56.422   Durbin-Watson:                   1.816
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               86.011
Skew:                          -1.294   Prob(JB):                     2.10e-19
Kurtosis:                       2.964   Cond. No.                         7.72
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
means for REGION by cluster
         REGION_x
cluster          
0        2.529412
1        2.581633
2        3.000000
standard deviations for REGION by cluster
         REGION_x
cluster          
0        0.796617
1        0.716858
2        0.000000
Multiple Comparison of Means - Tukey HSD,FWER=0.05
============================================
group1 group2 meandiff  lower  upper  reject
--------------------------------------------
  0      1     0.0522  -0.1693 0.2738 False 
  0      2     0.4706  -0.2761 1.2173 False 
  1      2     0.4184  -0.3398 1.1765 False 
--------------------------------------------

</code></pre>
