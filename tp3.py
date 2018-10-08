from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances

iris = datasets.load_iris()
x = iris.data
y = iris.target




def ppv(x, y, voisin):
    r = []
    neight = KNeighborsClassifier(n_neighbors=voisin)
    neight.fit(x,y)
    for data in x:
        p = metrics.pairwise.euclidean_distances(data, x)
        kpp = x[np.where(np.argsort(p) == 1)[1]]
        r.append(neight.predict(kpp)[0])
    return r        


print ppv(x, y, 3)