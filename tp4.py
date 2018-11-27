from sklearn import datasets
import numpy as np
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

pca = PCA(n_components=2)

x = datasets.load_iris().data
y = datasets.load_iris().target
dataX = pca.fit_transform(x)


def getBarycenter(x, y, k):
        if (y == k).sum():
                return x[np.where(np.column_stack((x,y))[:,4] == k)].mean(0)
        else :
                return np.zeros(x.shape[1])
        
def k_mean(x, k):
        d = x[np.random.choice(x.shape[0], k, replace=False)]
        nb = []
        while(1):
                p = euclidean_distances(x, d)
                label = np.argmin(p, axis=1)
                d = np.array([getBarycenter(x, label, el) for el in range(k)])
                nb.append(d.sum())
                if len(nb) > 1 and nb[-2] == nb[-1] and nb[-3] == nb[-2]:
                        break
        return label, nb

label, n = k_mean(x, 3)
plt.figure()
plt.subplot(2, 1, 1)
plt.title("Mon K-Mean")
plt.scatter(dataX[:, 0], dataX[:, 1], c=label)
labelSCA = pca.fit_transform([getBarycenter(x, label, el) for el in np.unique(label)])



kmeans = KMeans(n_clusters=3, random_state=0).fit(x)

plt.subplot(2, 1, 2)
plt.title("K-Mean de Sklearn")
plt.scatter(dataX[:, 0], dataX[:, 1], c=kmeans.labels_)
plt.show()


def best_K(x, k):
        y, n = k_mean(x, k)
        barycenters = np.unique(np.column_stack((y, np.array([getBarycenter(x, y, el) for el in y]))), axis=0)
        r = []
        for ligne in barycenters:
                r.append(euclidean_distances([ligne[1:]],x[np.where(np.column_stack((x,y))[:,-1] == ligne[0])], squared=True).sum())
        return np.array(r).sum()

plt.figure("Best K-mean")
nbK = [best_K(x, i) for i in range(2,10)]
plt.plot([x for x in range(2,10)], nbK)
plt.show()

print(metrics.silhouette_score(x, label, metric='euclidean'))
