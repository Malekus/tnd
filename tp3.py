import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()

a = np.array([1,2,3,4,5,6,7,8,9])

x = iris.data
y = iris.target

data = np.column_stack((x,y))

def PPV(x, y):
        r = np.array([y[np.where(np.argsort(euclidean_distances([i],x)) == 1)[1]][0] for i in x])
        return r, np.sum(r == y) / len(y)

def KNN(x, y, k):
        neigh = KNeighborsClassifier(n_neighbors=k)       
        neigh.fit(x, y)
        return neigh.predict(x), neigh.score(x, y)


def getBarycenter(x, y, k):
        if (y == k).sum():
                return x[y==k].mean(0)
        else :
                return None

def barycenters(x, y):
    return np.array([getBarycenter(x, y, k) for k in np.unique(y)])

def probClasse(x, y):
        r = []
        for i, j in zip(euclidean_distances(x, barycenters(x, y)), euclidean_distances(x, barycenters(x, y)).sum(axis=1)) :
                r.append(1 - i / j)
        return np.array(r)


def CBN(x, y):
        r = np.argmax(probClasse(x, y) * [(y == i).sum() / len(y) for i in np.unique(y)], axis=1)
        return r, round((r == y).sum() / len(x) * 100)
    

print(CBN(x, y))

gauss = GaussianNB()
gauss.fit(x, y)
gx = gauss.predict(x)
print(accuracy_score(gx, y))
