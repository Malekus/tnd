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
