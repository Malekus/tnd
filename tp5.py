import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


C = np.array(np.genfromtxt('choixprojetstab.csv',
                              dtype='str', delimiter=';', skip_header=1, usecols=[0]))
M = np.array(np.genfromtxt('choixprojetstab.csv',
                              dtype=float, delimiter=';', skip_header=1, usecols=range(1, 81)))

color = ['red','c','green', 'purple', 'yellow']
"""
Find best n clusters



"""

monK = 3


bayes = BayesianGaussianMixture(n_components=monK).fit(M)
kmean = KMeans(n_clusters=monK, random_state=0).fit(M)
spectral = SpectralClustering(n_clusters=monK, random_state=0).fit(M)
agglo = AgglomerativeClustering(n_clusters=monK).fit(M)

plt.figure()
plt.subplot(2,2,1)
plt.title("Bayes")
plt.scatter(PCA(n_components=2).fit_transform(M)[:,0], PCA(n_components=2).fit_transform(M)[:,1],
            c=np.array(color)[bayes.fit_predict(M)], alpha=0.8)
plt.subplot(2,2,2)
plt.title("Kmean")
plt.scatter(PCA(n_components=2).fit_transform(M)[:,0], PCA(n_components=2).fit_transform(M)[:,1],
            c=np.array(color)[kmean.fit_predict(M)], alpha=0.8)
plt.subplot(2,2,3)
plt.title("Spectral")
plt.scatter(PCA(n_components=2).fit_transform(M)[:,0], PCA(n_components=2).fit_transform(M)[:,1],
            c=np.array(color)[spectral.fit_predict(M)], alpha=0.8)
plt.subplot(2,2,4)
plt.title("Agglo")
plt.scatter(PCA(n_components=2).fit_transform(M)[:,0], PCA(n_components=2).fit_transform(M)[:,1],
            c=np.array(color)[agglo.fit_predict(M)], alpha=0.8)
plt.show()


"""

print("Affichage")
print(bayes.fit_predict(M))
print(kmean.fit_predict(M))
print(spectral.fit_predict(M))
print(agglo.fit_predict(M))
"""



a = bayes.fit_predict(M)
b = kmean.fit_predict(M)
c = spectral.fit_predict(M)
d = agglo.fit_predict(M)

r = np.array([[i,j,k,l] for i,j,k,l in zip(a,b,c,d)])
abcd = np.array([[np.sum(ij == ji) for ij in r] for ji in r])


matrix_simi = cosine_similarity(r)
cltEnd = SpectralClustering(n_clusters=monK, random_state=0, affinity='precomputed')
cltEnd.fit(matrix_simi)
plt.figure("Resultat de fin")
plt.scatter(PCA(n_components=2).fit_transform(M)[:,0], PCA(n_components=2).fit_transform(M)[:,1],
            c=np.array(color)[cltEnd.fit_predict(matrix_simi)], alpha=0.8)
plt.show()

