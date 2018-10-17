from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.naive_bayes import GaussianNB
#  np.set_printoptions(threshold=np.nan)
iris = datasets.load_iris()
x = iris.data
y = iris.target


def ppv(x, y, voisin):
    r = []
    neight = KNeighborsClassifier(n_neighbors=voisin)
    neight.fit(x,y)
    for data in x:
        p = metrics.pairwise.euclidean_distances(data.reshape(1,-1), x)
        kpp = x[np.where(np.argsort(p) == 1)[1]]
        r.append(neight.predict(kpp)[0])
    
    
    return r, round((1 - (sum(iris.target == r) / float(iris.target.shape[0]))) * 100, 1)


predictData, taux =  ppv(x, y, 1)

print predictData, taux


"""
def CBN(x, y):
    r = []
    clf = GaussianNB
    clf.fit(x, y)
    for data in x:
        print data
        


for k in np.unique(iris.target):
    k_len =  np.where(iris.target==k)
    print iris.data[k_len,:].mean(1)
    #print iris.data[k_len,:].mean(1)
    #print np.asarray(np.where(iris.target==k)).size
"""     