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
data, label = datasets.make_blobs(n_samples=500, n_features=2, centers=4)




#creer une figure vide
plt.figure()

#definir les limites en x [-15,15]
plt.xlim(-15, 15)

#definir les limites en y [-15,15]
plt.ylim(-15, 15)

#utilisation du scatter
plt.scatter(x=[x for index, x in enumerate(data[:,0]) if label[index] == 0],
            y=[y for index, y in enumerate(data[:,1]) if label[index] == 0], label="rouge", color="r")
plt.scatter(x=[x for index, x in enumerate(data[:,0]) if label[index] == 1],
            y=[y for index, y in enumerate(data[:,1]) if label[index] == 1], label="bleu", color="b")
plt.scatter(x=[x for index, x in enumerate(data[:,0]) if label[index] == 2],
            y=[y for index, y in enumerate(data[:,1]) if label[index] == 2], label="vert", color="g")
plt.scatter(x=[x for index, x in enumerate(data[:,0]) if label[index] == 3],
            y=[y for index, y in enumerate(data[:,1]) if label[index] == 3], label="jaune", color="y")

#titre du plot
plt.title("1000 donnees de deux variables\nreparties en 4 groupes")

#definir un label pour x
plt.xlabel("valeur de x")

#definir un label pour y
plt.ylabel("valeur de y")

#afficher la legende sur le graphe
plt.legend()

#afficher le graphe
plt.show()

a, b = datasets.make_blobs(n_samples=50, n_features=2, centers=2)
c, d = datasets.make_blobs(n_samples=250, n_features=2, centers=3)

plt.figure()
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(x=[x for index, x in enumerate(a[:,0]) if b[index] == 0],
            y=[y for index, y in enumerate(a[:,1]) if b[index] == 0],
            label="rouge", color="r")
plt.scatter(x=[x for index, x in enumerate(a[:,0]) if b[index] == 1],
            y=[y for index, y in enumerate(a[:,1]) if b[index] == 1],
            label="bleu", color="b")
plt.title("100 donnees de deux variables\nreparties en 2 groupes")
plt.xlabel("valeur de x")
plt.ylabel("valeur de y")
plt.legend()
plt.show()

plt.figure()
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(x=[x for index, x in enumerate(c[:,0]) if d[index] == 0],
            y=[y for index, y in enumerate(c[:,1]) if d[index] == 0],
            label="rouge", color="r")
plt.scatter(x=[x for index, x in enumerate(c[:,0]) if d[index] == 1],
            y=[y for index, y in enumerate(c[:,1]) if d[index] == 1],
            label="bleu", color="b")
plt.scatter(x=[x for index, x in enumerate(c[:,0]) if d[index] == 2],
            y=[y for index, y in enumerate(c[:,1]) if d[index] == 2],
            label="vert", color="g")
plt.title("500 donnees de deux variables\nreparties en 3 groupes")
plt.xlabel("valeur de x")
plt.ylabel("valeur de y")
plt.legend()
plt.show()

#concatenation des deux datasets concatenation honrizontal hstack (n,)
#sinon concatenation vertival vstack(n,m)
lastData, lastLabel = np.vstack((a, c)), np.hstack((b, d))

plt.figure()
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(x=[x for index, x in enumerate(lastData[:,0]) if lastLabel[index] == 0],
            y=[y for index, y in enumerate(lastData[:,1]) if lastLabel[index] == 0],
            label="rouge", color="r")
plt.scatter(x=[x for index, x in enumerate(lastData[:,0]) if lastLabel[index] == 1],
            y=[y for index, y in enumerate(lastData[:,1]) if lastLabel[index] == 1],
            label="bleu", color="b")
plt.scatter(x=[x for index, x in enumerate(lastData[:,0]) if lastLabel[index] == 2],
            y=[y for index, y in enumerate(lastData[:,1]) if lastLabel[index] == 2],
            label="vert", color="g")
plt.title("Concatenation des donnees")
plt.xlabel("valeur de x")
plt.ylabel("valeur de y")
plt.legend()
plt.show()
