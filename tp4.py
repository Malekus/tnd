from sklearn import datasets
import numpy as np
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

x = datasets.load_iris().data
y = datasets.load_iris().target
dataX = pca.fit_transform(x)
def k_moyenne(x, k, y = []):
        if len(y) == 0:
                p = np.random.choice(x.shape[0], k, replace=False)
                g = np.array([ x[i] for i in p])
                return g, p, np.array([np.argmin(euclidean_distances([i], g)) for i in x])
        else:
                p = [np.random.choice(np.where(c == i)[0]) for i in range(0, k)]
                g = np.array([ x[i] for i in p])
                return g, p, np.array([np.argmin(euclidean_distances([i], g)) for i in x])

def total(x, y, k, pK):
        return np.array([1 / sum(y.reshape(-1,1) == i) * euclidean_distances([pK[i]], x[np.where(np.column_stack((x,y))[:,4] == i)]).sum() for i in range(0, k)]).sum()

def rk(x, k, maxa):
        dx = []
        dy = []
        plt.figure("K-mean")
        for i in range(100):
                if i == 0:
                        a,b,c = k_moyenne(x, k)
                        d = c
                        p = total(x, c, k, a)
                        dy.append(x.shape[0] - sum(d==c))
                        dx.append(i)
                else:
                        d = c
                        a,b,c = k_moyenne(x, k, c)
                        p = total(x, c, k, a)
                        dy.append(x.shape[0] - sum(d==c))
                        dx.append(i)

        d = c
        plt.scatter(x=maxa[0], y=maxa[1], label=c)
        plt.show()
        return 

a,b,c = k_moyenne(x, 5)

rk(x, 5, dataX)
