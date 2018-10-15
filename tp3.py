from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances

iris = datasets.load_iris()
x = iris.data
y = iris.target




def ppv(x, y, K):
    r = []
    neight = KNeighborsClassifier(n_neighbors=K)
    neight.fit(x,y)
    for data in x:
        p = metrics.pairwise.euclidean_distances(data.reshape(1,-1), x)
        kpp = x[np.where(np.argsort(p) == 1)[1]]
        r.append(neight.predict(kpp)[0])
    
    
    return r, round((1 - (sum(iris.target == r) / float(iris.target.shape[0]))) * 100, 1)


predictData, taux =  ppv(x, y, 1)



print taux