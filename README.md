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

<table>
  <thead>
    <tr>
      <th>Month</th>
      <th>Savings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>January</td>
      <td>$100</td>
    </tr>
    <tr>
      <td>February</td>
      <td>$80</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td>Sum</td>
      <td>$180</td>
    </tr>
  </tfoot>
</table>
