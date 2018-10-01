# -*- coding: utf-8 -*-
from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA

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
            plt.scatter(x=iris.data[:,ligne], y=iris.data[:,colonne], c=iris.target, alpha=0.8)
            plt.title("Figure : Ligne "+ str(ligne) + " Colonne " + str(colonne))

plt.show()
#Figure Ligne 1 Colonne 3 ou Figure Ligne 0 Colonne 3

pca = PCA(n_components=2)

irisPCA = pca.fit(iris.data).transform(iris.data)
irisLDA = LDA().fit(iris.data, iris.target).transform(iris.data)

plt.figure("Figure PCA/LDA de iris data")
plt.subplot(1, 3, 1)
plt.scatter(irisPCA[:,0], irisPCA[:,1], c=iris.target, alpha=0.8)
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(irisLDA[:,0], irisLDA[:,1], c=iris.target, alpha=0.8)
plt.subplot(1, 3, 3)
plt.scatter(irisLDA[:,0], irisLDA[:,1], c=LDA().fit(iris.data, iris.target).predict(iris.data), alpha=0.8)
plt.legend()
plt.show()

print("Le taux de réussite est de " +  str(sum(LDA().fit(iris.data, iris.target).predict(iris.data) == iris.target) / (iris.data.shape[0] + 0.0)))
