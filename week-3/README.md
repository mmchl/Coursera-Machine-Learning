<h1>Coursera Machine Learning (Week 3)</h1>
<h2>Dataset information</h2>
<p><a href="https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring">https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring</a></p>
<p>The data set refers to Oxford Parkinson's Disease Telemonitoring.</p>
<p>Number of observations: 5875</p>
<p>Attributes: 26</p>
<ul>
  <li>subject - Integer that uniquely identifies each subject</li>
  <li>age - Subject age</li>
<li>sex - Subject gender '0' - male, '1' - female </li>
<li>test_time - Time since recruitment into the trial. The integer part is the number of days since recruitment. </li>
<li>motor_UPDRS - Clinician's motor UPDRS score, linearly interpolated </li>
<li>total_UPDRS - Clinician's total UPDRS score, linearly interpolated </li>
<li>Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP - Several measures of variation in fundamental frequency </li>
<li>Shimmer,Shimmer(DB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA - Several measures of variation in amplitude </li>
<li>NHR,HNR - Two measures of ratio of noise to tonal components in the voice </li>
<li>RPDE - A nonlinear dynamical complexity measure </li>
<li>DFA - Signal fractal scaling exponent </li>
<li>PPE - A nonlinear measure of fundamental frequency variation </li>
  </ul>
<h2>Assignment</h2>
<p>The assignment is to run a lasso regression analysis using k-fold cross validation to identify a subset of predictors from a larger pool of predictor variables that best predicts a quantitative response variable.</p>
<h2>Results</h2>
<pre><code>
In [24]: import pandas as pd
    ...: import numpy as np
    ...: import matplotlib.pylab as plt
    ...: from sklearn.cross_validation import train_test_split
    ...: from sklearn.linear_model import LassoLarsCV
    ...: 
    ...: #Load the dataset
    ...: data = pd.read_csv('parkinsons_updrs.csv')
    ...: 
    ...: #upper-case all DataFrame column names
    ...: data.columns = map(str.upper, data.columns)
    ...: 
    ...: # Data Management
    ...: data_clean = data.dropna()
    ...: 

In [25]: data_clean.dtypes
Out[25]: 
SUBJECT                int64
AGE                    int64
SEX                    int64
TEST_TIME            float64
MOTOR_UPDRS          float64
TOTAL_UPDRS          float64
JITTER_PERCENTAGE    float64
JITTER_ABS           float64
JITTER_RAP           float64
JITTER_PPQ5          float64
JITTER_DDP           float64
SHIMMER              float64
SHIMMER_DB           float64
SHIMMER_APQ3         float64
SHIMMER_APQ5         float64
SHIMMER_APQ11        float64
SHIMMER_DDA          float64
NHR                  float64
HNR                  float64
RPDE                 float64
DFA                  float64
PPE                  float64
dtype: object

In [26]: data_clean.describe()
Out[26]: 
           SUBJECT          AGE          SEX    TEST_TIME  MOTOR_UPDRS  \
count  5875.000000  5875.000000  5875.000000  5875.000000  5875.000000   
mean     21.494128    64.804936     0.317787    92.863722    21.296229   
std      12.372279     8.821524     0.465656    53.445602     8.129282   
min       1.000000    36.000000     0.000000    -4.262500     5.037700   
25%      10.000000    58.000000     0.000000    46.847500    15.000000   
50%      22.000000    65.000000     0.000000    91.523000    20.871000   
75%      33.000000    72.000000     1.000000   138.445000    27.596500   
max      42.000000    85.000000     1.000000   215.490000    39.511000   

       TOTAL_UPDRS  JITTER_PERCENTAGE   JITTER_ABS   JITTER_RAP  JITTER_PPQ5  \
count  5875.000000        5875.000000  5875.000000  5875.000000  5875.000000   
mean     29.018942           0.006154     0.000044     0.002987     0.003277   
std      10.700283           0.005624     0.000036     0.003124     0.003732   
min       7.000000           0.000830     0.000002     0.000330     0.000430   
25%      21.371000           0.003580     0.000022     0.001580     0.001820   
50%      27.576000           0.004900     0.000035     0.002250     0.002490   
75%      36.399000           0.006800     0.000053     0.003290     0.003460   
max      54.992000           0.099990     0.000446     0.057540     0.069560   

          ...        SHIMMER_DB  SHIMMER_APQ3  SHIMMER_APQ5  SHIMMER_APQ11  \
count     ...       5875.000000   5875.000000   5875.000000    5875.000000   
mean      ...          0.310960      0.017156      0.020144       0.027481   
std       ...          0.230254      0.013237      0.016664       0.019986   
min       ...          0.026000      0.001610      0.001940       0.002490   
25%       ...          0.175000      0.009280      0.010790       0.015665   
50%       ...          0.253000      0.013700      0.015940       0.022710   
75%       ...          0.365000      0.020575      0.023755       0.032715   
max       ...          2.107000      0.162670      0.167020       0.275460   

       SHIMMER_DDA          NHR          HNR         RPDE          DFA  \
count  5875.000000  5875.000000  5875.000000  5875.000000  5875.000000   
mean      0.051467     0.032120    21.679495     0.541473     0.653240   
std       0.039711     0.059692     4.291096     0.100986     0.070902   
min       0.004840     0.000286     1.659000     0.151020     0.514040   
25%       0.027830     0.010955    19.406000     0.469785     0.596180   
50%       0.041110     0.018448    21.920000     0.542250     0.643600   
75%       0.061735     0.031463    24.444000     0.614045     0.711335   
max       0.488020     0.748260    37.875000     0.966080     0.865600   

               PPE  
count  5875.000000  
mean      0.219589  
std       0.091498  
min       0.021983  
25%       0.156340  
50%       0.205500  
75%       0.264490  
max       0.731730  

[8 rows x 22 columns]
                     
In [34]: predvar = data_clean[['AGE','SEX', 'TEST_TIME', 'MOTOR_UPDRS', 'TOTAL_UPDRS',
    ...:                       'JITTER_PERCENTAGE', 'JITTER_ABS',
    ...:                       'JITTER_PPQ5', 'JITTER_DDP', 'SHIMMER', 'SHIMMER_DB',
    ...:                       'SHIMMER_APQ3', 'SHIMMER_APQ5', 'SHIMMER_APQ11',
    ...:                       'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']]
    ...:                       

In [35]: target = data_clean.SUBJECT
    ...: 
    ...: # standardize predictors to have mean=0 and sd=1
    ...: predictors=predvar.copy()
    ...: from sklearn import preprocessing
    ...: for column in predictors.columns.tolist():
    ...:     predictors[column] = preprocessing.scale(predictors[column].astype('float64'))
    ...: 
    ...: # split data into train and test sets
    ...: pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target,
    ...:                                                               test_size=.3, random_state=15)
    ...: 
    ...: # specify the lasso regression model
    ...: model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
    ...: 

In [36]: # print variable names and regression coefficients
    ...: dict(zip(predictors.columns, model.coef_))
Out[36]: 
{'AGE': -1.5705996930289794,
 'DFA': 2.2428847834613475,
 'HNR': -1.0937141427459955,
 'JITTER_ABS': -1.7549704491278355,
 'JITTER_DDP': -4.8130542441192272,
 'JITTER_PERCENTAGE': 5.4235947698502516,
 'JITTER_PPQ5': -1.1886097517404706,
 'MOTOR_UPDRS': -1.5504633397681769,
 'NHR': 2.4734808044459244,
 'PPE': 0.21794341454730592,
 'RPDE': 0.73649924278729739,
 'SEX': 4.1978646513571576,
 'SHIMMER': 8.4893600901597832,
 'SHIMMER_APQ11': 3.1560985182727626,
 'SHIMMER_APQ3': -0.79450494905063318,
 'SHIMMER_APQ5': -7.4583815903468587,
 'SHIMMER_DB': -4.1242788648126911,
 'TEST_TIME': -0.030031022402810938,
 'TOTAL_UPDRS': 5.4065585205477369}

In [37]: # plot coefficient progression
    ...: m_log_alphas = -np.log10(model.alphas_)
    ...: ax = plt.gca()
    ...: plt.plot(m_log_alphas, model.coef_path_.T)
    ...: plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
    ...:             label='alpha CV')
    ...: plt.ylabel('Regression Coefficients')
    ...: plt.xlabel('-log(alpha)')
    ...: plt.title('Regression Coefficients Progression for Lasso Paths')
    ...:             
Out[37]: Text(0.5,1,'Regression Coefficients Progression for Lasso Paths')
￼</code></pre>
<img src="https://github.com/mmchl/Coursera-Machine-Learning/blob/master/week-3/plot-coefficient-progression.png">
<pre><code>
In [38]: m_log_alphascv = -np.log10(model.cv_alphas_)
    ...: plt.figure()
    ...: plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
    ...: plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
    ...:          label='Average across the folds', linewidth=2)
    ...: plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
    ...:             label='alpha CV')
    ...: plt.legend()
    ...: plt.xlabel('-log(alpha)')
    ...: plt.ylabel('Mean squared error')
    ...: plt.title('Mean squared error on each fold')
    ...:             
</code></pre>
<img src="https://github.com/mmchl/Coursera-Machine-Learning/blob/master/week-3/plot-mean-square-error.png">
<pre><code>
In [39]: from sklearn.metrics import mean_squared_error
    ...: train_error = mean_squared_error(tar_train, model.predict(pred_train))
    ...: test_error = mean_squared_error(tar_test, model.predict(pred_test))
    ...: print ('training data MSE')
    ...: print(train_error)
    ...: print ('test data MSE')
    ...: print(test_error)
    ...: 
</code></pre>
<p>training data MSE<br>
111.508292223<br>
test data MSE<br>
120.023088959</p>
<pre><code>
In [40]: rsquared_train=model.score(pred_train,tar_train)
    ...: rsquared_test=model.score(pred_test,tar_test)
    ...: print ('training data R-square')
    ...: print(rsquared_train)
    ...: print ('test data R-square')
    ...: print(rsquared_test)
    ...: 
</code></pre>
<p>training data R-square<br>
0.267879109649<br>
test data R-square<br>
0.224502789509</p>
