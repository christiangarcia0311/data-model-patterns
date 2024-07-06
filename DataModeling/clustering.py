# Clustering Model 
# Clusters 
# Data Modeling 
# Python Library 
# Author: Christian Garcia

# Import required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Heirarchical Clustering 
# Agglomerative method
class HeirarchicalClustering:
  def __init__(self, method='ward'):
    self.method = method 
    self.linkage_matrix = None 
    self.cluster_labels = None 
  
  def fit(self, X):
    
    # Perform heirarchical/agglomerative method clustering
    self.linkage_matrix = linkage(X, method=self.method)
    return self
  
  # __runnable
  # Return cluster dendogram 
  def dendrogram(self, style='fivethirtyeight'):
    plt.style.use(style)
    plt.figure(figsize=(10, 7))
    dendrogram(self.linkage_matrix)
    plt.title('heirarchical Clustering')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()
  
  # __runnable
  # Get cluster labels 
  def labels(self, n_clusters=3):
    self.cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
    return self.cluster_labels

# Partitive Clustering -> Kmeans 
class KMeansClustering:
  def __init__(self, n_clusters=3, random_state=0):
    self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    self.centers_ = None 
    self.labels_ = None 
    self.inertia_ = None
    self.best_K = None 
  
  def fit(self, X):
    
    # Train model 
    self.model.fit(X)
    
    # Get cluster centers, labels and inertia 
    self.centers_ = self.model.cluster_centers_
    self.labels_ = self.model.labels_
    self.inertia_ = self.model.inertia_ 
    
    return self
  
  # __runnable
  # Get cluster centers
  def centers(self, feature_names):
    cluster_centers = pd.DataFrame(self.centers_, columns=feature_names)
    cluster_centers.index.names = ['Clusters']
    return cluster_centers
  
  # __runnable
  # Get inertia 
  def inertia(self):
    return self.inertia_
  
  # __runnable  
  # Get cluster labels
  def labels(self):
    return self.labels_
  
  # __runnable
  # Plot clusters 
  def plot_clusters(self, X, xlabel='X-Axis', ylabel='Y-Axis', style='fivethirtyeight'):
    plt.style.use(style)
    plt.figure(figsize=(10, 7))
    
    # Plot each cluster in different color
    unique_labels = np.unique(self.labels_)
    
    for label in unique_labels:
      cluster_points = X[self.labels_ == label]
      plt.scatter(cluster_points[:,0], cluster_points[:, 1], label=f'Cluster {label + 1}')
    
    plt.scatter(self.centers_[:,0], self.centers_[:,1], s=100, c='red', marker='+', label='Centroids')
    plt.title('KMeans Clustering')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
  
  # __runnable
  # Get silhouette score of each number of Clusters
  def score(self, X, range_length=10, random_state=0):
    K = range(2, range_length)
    max_score = -1
    scores = []
    
    for k in K:
      model = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
      model.fit(X)
      label = model.fit_predict(X)
      score = silhouette_score(X, label, metric='euclidean').round(3)
      scores.append(score)
      
      if score > max_score:
        max_score = score 
        self.best_K = k 
    
    return pd.DataFrame({
      'Clusters': K,
      'Silhouette Score': scores
    })