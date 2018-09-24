# -*- coding: utf-8 -*-
from sklearn import *
import numpy as np
import matplotlib.pyplot as plt

#creation de la matrix x
x = np.array([[1,-1,2], [2,0,0], [0,1,-1]])

print x

print "moyenne de x est " + str(x.mean(0))

print "la variance de x est " + str(x.var(0))

ppx = preprocessing.scale(x)

print ppx
#les valeurs obtenus sont proches de 0

print ppx.mean(0), ppx.var(0)
#la moyenne de la matrice normaliser est 0 car la moyenne d'un maxtrice normaliser est tjrs 0
#la variance de la matrice normaliser est 1 car la moyenne d'un maxtrice normaliser est tjrs 1

x2 = np.array([[1, -1, 2], [2, 0, 0], [0, 1,-1]])
print x2
scaler = preprocessing.MinMaxScaler()
ppx2 = scaler.fit(x2)

print "toto"


# calculer la valeur min revient a calculer le moyenne de chaque variable
# calculer la valeur max revient a calculer le varia,ce de chaque variable
print np.min(scaler.transform(x2), axis=0)
print np.max(scaler.transform(x2), axis=0)

iris = datasets.load_iris()

m = 0
plt.figure(1)
for ligne in range(0, len(iris.feature_names)):
    for colonne in range(0, len(iris.feature_names)):
        if ligne != colonne and ligne <= colonne: 
            m = m + 1     
            plt.subplot(2, 3, m)
            plt.scatter(x=iris.data[:,ligne], y=iris.data[:,colonne], c=iris.target)
            plt.title("Ligne "+ str(ligne) + " Colonne " + str(colonne))
        

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
print pca.fit(iris.data).transform(iris.data).shape
plt.figure(2)
plt.scatter(pca.fit(iris.data).transform(iris.data)[:,0], pca.fit(iris.data).transform(iris.data)[:,1], c=iris.target)
plt.show()


