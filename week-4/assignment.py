from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

# Loading the dataset
data = pd.read_csv('wholesale.csv')
# Mapping Columns
data.columns = map(str.upper, data.columns)

data_clean = data.dropna()

cluster=data_clean
cluster.describe()

# Dict
regions_replacer = {'Other': 3, 'Lisbon': 1, 'Oporto': 2}
data_clean.REGION = data_clean.REGION.replace(regions_replacer)

data_clean.head()

clustervar = cluster.copy()
# Cycle for converting rows to float64
for col in clustervar.columns:
    clustervar[col] = preprocessing.scale(clustervar[col].astype('float64'))

clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

from scipy.spatial.distance import cdist
clusters=range(1,20)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
    / clus_train.shape[0])

'''
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
'''
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
# Show the plot
plt.show()

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)

# plot clusters
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

'''
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
'''
# create a unique identifier variable from the index for the
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
print(newlist)

# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
print(newclus)
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the
# cluster assignment dataframe
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()
'''
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
'''

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)

region_data=data_clean['REGION']
class_train, gpa_test = train_test_split(region_data, test_size=.3, random_state=123)
class_train1 = pd.DataFrame(class_train)
class_train1.reset_index(level=0, inplace=True)
merged_train_all = pd.merge(class_train1, merged_train, on='index')
print(merged_train_all)

sub1 = merged_train_all[['REGION_x', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

gpamod = smf.ols(formula='REGION_x ~ C(cluster)', data=sub1).fit()
print (gpamod.summary())

print ('means for REGION by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for REGION by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['REGION_x'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())
