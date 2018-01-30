import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

#Load the dataset
data = pd.read_csv('parkinsons_updrs.csv')

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)

# Data Management
data_clean = data.dropna()
data_clean.dtypes
data_clean.describe()

#select predictor variables and target variable as separate data sets
predvar = data_clean[['AGE','SEX', 'TEST_TIME', 'MOTOR_UPDRS', 'TOTAL_UPDRS',
                      'JITTER_PERCENTAGE', 'JITTER_ABS',
                      'JITTER_PPQ5', 'JITTER_DDP', 'SHIMMER', 'SHIMMER_DB',
                      'SHIMMER_APQ3', 'SHIMMER_APQ5', 'SHIMMER_APQ11',
                      'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']]


target = data_clean.SUBJECT

# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
from sklearn import preprocessing
for column in predictors.columns.tolist():
    predictors[column] = preprocessing.scale(predictors[column].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target,
                                                              test_size=.3, random_state=15)

# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')


# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
