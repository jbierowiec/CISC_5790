# Problem 2

# part a: Build a dendogram for this dataset using single-link, bottom-up approach. Show your work.

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

X = [[i] for i in [0, 4, 5, 20, 25, 39, 43, 44]]
X_arr = np.array(X)
print(X_arr)

n = X_arr.shape[0]
p = X_arr.shape[1]
dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        s = 0
        for k in range(p):
            s += (X_arr[i, k] - X_arr[j, k])**2
        dist[i, j] = np.sqrt(s)
dist

print(dist)

Z = linkage(X, 'single', 'euclidean')
fig = plt.figure(figsize=(25, 10))

labels = ['0', '4', '5', '20', '25', '39', '43', '44']
dendrogram = dendrogram(Z, labels=labels)
plt.title('Hierarchical Clustering Dendogram (Single-link Bottom-up)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.show()

# part b: Suppose we want the two top-level clusters. List the data points in each cluster.

# top level Cluster 1: [0, 4, 5]
# top level Cluster 2: [20, 25, 39, 43, 44]