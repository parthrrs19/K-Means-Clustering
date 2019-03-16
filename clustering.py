import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

college = pd.read_csv('College_Data',index_col=0)
print(college.head())
college.info()
print(college.describe())

#exploratory data analysis
sns.set()
sns.lmplot('Room.Board','Grad.Rate',data=college,hue='Private',scatter_kws={"s": 10},fit_reg=False)
plt.show()

sns.lmplot('Outstate','F.Undergrad',data=college,hue='Private',scatter_kws={"s": 10},fit_reg=False)
plt.show()

g = sns.FacetGrid(college,hue="Private",height=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
plt.show()

f = sns.FacetGrid(college,hue='Private',height=6,aspect=2)
f = f.map(plt.hist,'Grad.Rate',alpha=0.7,bins=20)
plt.show()

college[college['Grad.Rate']>100]
college.loc['Cazenovia College','Grad.Rate'] = 100
college[college['Grad.Rate']>100]
h = sns.FacetGrid(college,hue='Private',height=6,aspect=2)
h = h.map(plt.hist,'Grad.Rate',alpha=0.7,bins=20)
plt.show()

#k means cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(college.drop('Private',axis=1))
print(kmeans.cluster_centers_)

#evaluation
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
college['Cluster'] = college['Private'].apply(converter)
print(college.head())

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(college['Cluster'],kmeans.labels_))
print(classification_report(college['Cluster'],kmeans.labels_))
