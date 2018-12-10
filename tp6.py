import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

iris = datasets.load_iris()
x = datasets.load_iris().data
y = datasets.load_iris().target

class Analyse:
    def __init__(self, data):
        self.x = data.data
        self.y = data.target
        self.features = data.feature_names
        self.color = np.array(['royalblue', 'g', 'darkorange'])

    def analyse(self):
        print("Le nombre de donnees est " + str(self.x.size))
        print("Le nombre de variable est " + str(self.x.shape[1]))
        print("Les numeros de classes sont ")

    def vis(self):
        pca = PCA(n_components=2)
        lda = LDA(n_components=2)
        plt.figure("Visualisation")
        plt.subplot(1,2,1)
        plt.scatter(pca.fit_transform(x)[:,0],
                    pca.fit_transform(x)[:,1],
                    c=y)
        plt.subplot(1,2,2)
        plt.scatter(lda.fit(x,y).transform(x)[:,0],
                    lda.fit(x,y).transform(x)[:,1],
                    c=y)
        plt.show()

    def varVis(self):       
        m = 0
        plt.figure()
        for ligne in range(0, len(self.features)):
            for colonne in range(0, len(self.features)):
                if ligne != colonne:
                    m = m + 1
                    plt.subplot(len(self.features), len(self.features)-1, m)
                    plt.scatter(x=self.x[:,colonne], y=self.x[:,ligne], c=self.color[self.y], alpha=0.8)
                    plt.xlabel(self.features[colonne])
                    plt.ylabel(self.features[ligne])
        plt.show()

    def regression_lin(self):
        m = 0
        plt.figure()
        for ligne in range(0, len(self.features)):
            for colonne in range(0, len(self.features)):
                if ligne != colonne:
                    m = m + 1
                    plt.subplot(len(self.features), len(self.features)-1, m)
                    plt.scatter(x=self.x[:,colonne], y=self.x[:,ligne], c=self.color[self.y], alpha=0.8)
                    plt.xlabel(self.features[colonne])
                    plt.ylabel(self.features[ligne])                    
                    x, y = self.makeRegressionLinear(np.array([self.x[:,colonne]]).transpose(), self.x[:, ligne])
                    plt.plot(x,y, c='r')            
        plt.show()
        
    def makeRegressionLinear(self, x, y):
        reglin = LinearRegression()
        reglin.fit(x, y)
        r = list(sorted(x))
        return r, reglin.predict(r)
    
    
    def classification(self, k=3, pourcentage=0.4):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            test_size=pourcentage, 
                                                            random_state=0)
        KNN = KNeighborsClassifier(n_neighbors=k)
        res = []
        for index, data  in enumerate(self.x):
            KNN.fit(np.delete(self.x, index, axis=0), np.delete(self.y, index, axis=0))
            res.append(KNN.predict([data])[0])
        
        return res
        
        
a = Analyse(iris)
a.classification()
