#importatioon des librairies
from sklearn import *
import numpy as np
import matplotlib.pyplot as plt

#charger les donnees iris
iris = datasets.load_iris()

#afficher les donnees
print iris.data

#afficher les noms des variables
print iris.feature_names

#afficher les noms des classes
print iris.target_names

#afficher nom des classes pour chaque donnees
for index, row in enumerate(iris.data):
    print np.append(row, iris.target_names[iris.target[index]])

#afficher la moyenne pour chaque variable
print iris.data.mean(0)

#afficher l'ecart-type pour chaque variable
print iris.data.std(0)

#afficher min pour chaque variable
print iris.data.min(0)

#afficher max pour chaque variable
print iris.data.max(0)

#afficher le nombre de donnees
print iris.data.size

#afficher le nombre de variables
print iris.data.shape[0]

#afficher le nombre de classes
print iris.target_names.size

#importer des donnes depuis internet
mnist = datasets.fetch_mldata('MNIST original')

#afficher la matrix MNIST
print mnist.data

#afficher nombre de donnes MNIST
print mnist.data.size

#afficher le nombre de variable MNIST
print mnist.data.shape[0]

#afficher le numero de chaque classse


#afficher la moyenne pour chaque variable MNIST
print mnist.data.mean(0)

#afficher l'ecart-type pour chaque variable MNIST
print mnist.data.std(0)

#afficher min pour chaque variable MNIST
print mnist.data.min(0)

#afficher max pour chaque variable MNIST
print mnist.data.max(0)

#afficher le nombre de classe avec la fonction unique
print np.unique(mnist.target)

#utilisation de la fonction help sur datasets.make_blobs
#print help(datasets.make_blobs)

#generer des donnees 1000 donnes=n_samples, 2 variables=n_features, 4 groupes=centers
(data,label) = datasets.make_blobs(n_samples=1000, n_features=2, centers=4)
print data, label

plt.figure()
#plt.scatter(data.mean(0), label)

print data.shape[0], label.shape

#plt.show()
