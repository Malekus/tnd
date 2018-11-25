import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split


C = np.array(np.genfromtxt('choixprojetstab.csv',
                              dtype='str', delimiter=';', skip_header=1, usecols=[0]))
M = np.array(np.genfromtxt('choixprojetstab.csv',
                              dtype=float, delimiter=';', skip_header=1, usecols=range(1, 81)))

color = ['red','c','green']
"""
Find best n clusters
"""

x_train, x_test = train_test_split(M, test_size=0.3)

print(x_train, x_test)

bayes = BayesianGaussianMixture(n_components=3).fit(x_train)
kmean = KMeans(n_clusters=3, random_state=0).fit(x_train)
spectral = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit(x_train)
agglo = AgglomerativeClustering(n_clusters=3).fit(x_train)

print(bayes)
print(kmean)
print(spectral)
print(agglo)


plt.figure()
plt.subplot(2,2,1)
plt.title("Bayes")
plt.scatter(PCA(n_components=2).fit_transform(x_test)[:,0], PCA(n_components=2).fit_transform(x_test)[:,1],
            c=np.array(color)[bayes.fit_predict(x_test)], alpha=0.8)
plt.subplot(2,2,2)
plt.title("Kmean")
plt.scatter(PCA(n_components=2).fit_transform(x_test)[:,0], PCA(n_components=2).fit_transform(x_test)[:,1],
            c=np.array(color)[kmean.fit_predict(x_test)], alpha=0.8)
plt.subplot(2,2,3)
plt.title("Spectral")
plt.scatter(PCA(n_components=2).fit_transform(x_test)[:,0], PCA(n_components=2).fit_transform(x_test)[:,1],
            c=np.array(color)[spectral.fit_predict(x_test)], alpha=0.8)
plt.subplot(2,2,4)
plt.title("Agglo")
plt.scatter(PCA(n_components=2).fit_transform(x_test)[:,0], PCA(n_components=2).fit_transform(x_test)[:,1],
            c=np.array(color)[agglo.fit_predict(x_test)], alpha=0.8)
plt.show()



print("Affichage")
print(bayes.fit_predict(x_test))
print(kmean.fit_predict(x_test))
print(spectral.fit_predict(x_test))
print(agglo.fit_predict(x_test))