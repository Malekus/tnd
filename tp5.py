import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

C = np.array(np.genfromtxt('choixprojetstab.csv',
                              dtype='str', delimiter=';', skip_header=1, usecols=[0]))
M = np.array(np.genfromtxt('choixprojetstab.csv',
                              dtype=float, delimiter=';', skip_header=1, usecols=range(1, 81)))


pca = PCA(n_components=2)
scaler = StandardScaler()

T = scaler.fit_transform(M)

kmeans = KMeans(n_clusters=2, random_state=0).fit(M)


data = pca.fit_transform(T)
"""
plt.figure(1)
plt.scatter(data[:,0],data[:,1], c=kmeans.labels_)
plt.show()
"""
clustering = AffinityPropagation().fit(M)
clustering = AgglomerativeClustering(n_clusters=2).fit(T)

plt.figure(2)
plt.scatter(data[:,0],data[:,1], c=clustering.labels_)
#plt.plot(data[:,0],data[:,1])
plt.show()

"""
KMean
AffinityPropagation
AgglomerativeClustering
"""
