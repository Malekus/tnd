import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

a = np.array([1,2,3,4,5,6,7,8,9])

x = iris.data
y = iris.target

data = np.column_stack((x,y))

def getBarycenter(x, y, k):
        if (y == k).sum():
                return x[np.where(np.column_stack((x,y))[:,4] == k)].mean(0)
        else :
                return np.zeros(x.shape[1])


def barycenters(x, y):
    return np.array([getBarycenter(x, y, k) for k in np.unique(y)])


def probabiliteClasseK(data, label, x, k):
    return 1 - (euclidean_distances([x], [getBarycenter(data, label, k)])[0][0]
                / euclidean_distances([x], barycenters(data, label)).sum())


    
def probabiliteClasseXK(data, label, x, k):
    b = probabiliteClasseK(data, label, x, k)
    a = 1 / data.shape[0]
    return (a * b) / b


def classDes(data, label, x):
    a =  np.array([
        (probabiliteClasseXK(data, label, x, 0) * probabiliteClasseK(data, label, x, 0),0),
        (probabiliteClasseXK(data, label, x, 1) * probabiliteClasseK(data, label, x, 1),1),
        (probabiliteClasseXK(data, label, x, 2) * probabiliteClasseK(data, label, x, 2),2)
    ])

    return [p for k,p in a if k == np.max(a, axis=0)[0]][0]

def CBN(x, y):
    r = np.array([classDes(x, y, x1) for x1 in x])
    return r, round(1 - sum(y == r) / r.shape[0], 2)

print(CBN(x, y))

gauss = GaussianNB()
gauss.fit(x, y)

gaussPredict = [gauss.predict([x1]) for x1 in x]
print(gauss.score(gaussPredict, y))
