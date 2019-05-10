<h1>A Comparison of Clustering Algorithms (K-means MeanShift DBSCAN) in Python</h1>
This article compares 3 different clustering algorithms found in scikit-learn, Python's Machine Learning library.
The database used here can be found at Kaggle. You can download the database directly from <a href="https://www.kaggle.com/shwetabh123/mall-customers/downloads/mall-customers.zip/1">here</a>.
<h2>Clustering</h2>
Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). (<a href="https://en.wikipedia.org/wiki/Cluster_analysis".>Wikipedia</a>)<br>
Clustering is basically grouping observations based on given characteristics.<br>
Normally, clustering gives better results when used together with principal components; but for the sake of this comparison, things were kept simple - this example uses clustering algorithms directly on a database.
<h3>Database</h3>
The database used here is a customer database from a mall. It includes customer profile information and a previously determined spending score.
<h3>
