<h1>A Comparison of Clustering Algorithms (K-means MeanShift DBSCAN) in Python</h1>
This article compares 3 different clustering algorithms found in scikit-learn, Python's Machine Learning library. You'll be presented how these algorithms are used and optimized according to your needs.<br>
The database used here can be found at Kaggle. You can download the database directly from <a href="https://www.kaggle.com/shwetabh123/mall-customers/downloads/mall-customers.zip/1">here</a>.
<h2>Definition of Clustering</h2>
Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). (<a href="https://en.wikipedia.org/wiki/Cluster_analysis".>Wikipedia</a>)<br>
Clustering is basically grouping observations based on given characteristics.<br>
Normally, clustering provides better results when used together with principal components; but for the sake of this comparison, things were kept simple - this example uses clustering algorithms directly on the columns of the given database.
<h2>Database</h2>
The database used here is a customer database from a mall. It includes customer profile information and a previously determined spending score, which further makes things easy for a clustering analysis.
<h2>Analysis</h2>
Let's start by importing modules and the database into Python.<br><br>
<pre>import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/Emir/Desktop/Mall Customers.csv')
df.head()</pre><br>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
  </tbody>
</table>

<br>
A quick look at pairs of variables with scatter plot gives us what we need: Spending score and annual income make a perfect pair for clustering.
<pre>df.plot.scatter(x='Spending Score (1-100)', y='Annual Income (k$)',  figsize=(15,7))</pre>
<img src="https://github.com/EmirKorkutUnal/A-Comparison-of-Clustering-Algorithms-K-means-MeanShift-DBSCAN-in-Python/blob/master/Screenshots/1%20-%20ScatterClean.JPG">
We can already see the approximate groups in this plot:
<img src="https://github.com/EmirKorkutUnal/A-Comparison-of-Clustering-Algorithms-K-means-MeanShift-DBSCAN-in-Python/blob/master/Screenshots/2%20-%20ScatterApprox.jpg">
Let's see how scikit-learn's clustering algorithms will group these observations.<br>
<h3>K-means</h3>
K-means algorithm works by specifying a certain number of clusters beforehand.<br>
First we load the K-means module, then we create a database that only consists of the two variables we selected.<br>
<pre>from sklearn.cluster import KMeans
x = df.filter(['Annual Income (k$)','Spending Score (1-100)'])</pre>
Because we can obviously see that there are 5 clusters, we will force K-means to create exactly 5 clusters for us.<br>
<pre>kmeans = KMeans(n_clusters=5)</pre>
Now we can fit our data into the model.
<pre>clusters = kmeans.fit(x)</pre>
We can get the results by typing ".labels_" after the name of the model.
<pre>In[7]:  clusters.labels_
Out[7]: array([3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
        3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 0,
        3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 1, 4, 1, 4, 1,
        0, 1, 4, 1, 4, 1, 4, 1, 4, 1, 0, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
        4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
        4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
        4, 1])</pre>
Next step is turning this array into a dataframe and adding it to our original dataframe. We're also renaming the column that contains cluster numbers.
<pre>ClusterDataset = pd.DataFrame(data=clusters.labels_)
dfClustered = pd.concat([df, ClusterDataset], axis=1)
dfClustered.rename(columns={0:'Cluster'}, inplace=True)</pre>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
Time to get our first results. "c=" will color the groups according to the 'Cluster' column, "cmap=" will use a specified scheme for colorization.
<pre>dfClustered.plot.scatter(x='Spending Score (1-100)', y='Annual Income (k$)', c='Cluster', cmap="gist_rainbow", figsize=(15,7))</pre>
<img src="https://github.com/EmirKorkutUnal/A-Comparison-of-Clustering-Algorithms-K-means-MeanShift-DBSCAN-in-Python/blob/master/Screenshots/3%20-%20ScatterKmeans.JPG">
It looks pretty good; there are 3 observations one can argue that they would belong to the purple cluster rather than the red, also one observation could be classified within the green cluster instead of the red.<br>
Let's look at how other algorithms would do the job.
<h3>DBSCAN</h3>
DBSCAN creates clusters in a different way than K-means. "min_samples=" allows you to specify a minimum cluster size, and "eps=" is the maximum distance between two obsertavions for them to be considered within the same cluster. This approach allows a more flexible clustering operation, giving control from algorithm to the analyst. Based on these 2 inputs, DBSCAN can also identify some observations as outliers and not include them in any cluster. These observations are labeled with a cluster number of "-1".<br><br>
We will use the same code as above, changing only these 2 lines.
<pre>from sklearn.cluster import DBSCAN</pre>
<pre>dbscan = DBSCAN(eps=12, min_samples=10)</pre>
Remember to change the epsilon (eps) and minimum cluster size according to your own needs and run the code as many times as you need with some variation of numbers to get the optimum results. Eps can also be a <i>float</i> number, meaning that decimals are allowed for this variable.<br>
After some trial and error, using eps as 12 and min_samples as 10 gave a reasonable result.
<img src="https://github.com/EmirKorkutUnal/A-Comparison-of-Clustering-Algorithms-K-means-MeanShift-DBSCAN-in-Python/blob/master/Screenshots/4%20-%20ScatterDBSCAN.JPG">
<pre>dfClustered.groupby('Cluster').size()
Cluster
-1    28
 0    16
 1    10
 2    92
 3    31
 4    23
dtype: int64<pre>
As you can see, the smallest group really has only 10 observations, and 28 observations are not included in any of the groups.
