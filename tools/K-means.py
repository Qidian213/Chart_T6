import numpy as np
from sklearn.cluster import KMeans

X = np.array([2, 4, 0, 2.5, 4, 0])
              
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print(kmeans.labels_)
print(kmeans.predict([0, 3]))
print(kmeans.cluster_centers_)
