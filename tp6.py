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
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import warnings
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score



iris = datasets.load_iris()
wine = datasets.load_wine()
digits = datasets.load_digits()

x = datasets.load_iris().data
y = datasets.load_iris().target

class Analyse:
    def __init__(self, data):
        self.x = data.data
        self.y = data.target
        self.features = data.feature_names if 'feature_names' in data.keys() else ['Var_'+ str(k) for k in range(len(data.data[0]))]
        self.color = np.array(['#1f77b4', '#ff7f0e', '#2ca02c']) if len(np.unique(data.target)) <= 3 else None
        self.target_names = data.target_names if 'target_names' in data.keys() else None

    def analyse(self):
        print("Nombre de donnees : " + str(self.x.size))
        print("Nombre de variable : " + str(self.x.shape[1]))
        print("Numero de classe : " + str(np.unique(self.y)))
        for index, v in enumerate(self.features):
            print("Variable : " + v)
            print("\tMoyenne : " + str(round(self.x[:, index].mean(0),2)))
            print("\tEcart type : " + str(round(self.x[:, index].std(0),2)))
            print("\tMin : " + str(round(self.x[:, index].min(0),2)))
            print("\tMax : " + str(round(self.x[:, index].max(0),2)))
        
    def vis(self):
        warnings.simplefilter("ignore")
        pcaData = PCA(n_components=2).fit_transform(self.x)
        ldaData = LDA(n_components=2).fit(self.x, self.y).transform(self.x)
    
        plt.figure("Visualisation des donnees PCA et LDA")
        plt.subplot(1,2,1)        
        plt.title("PCA")
        plt.scatter(pcaData[:,0], pcaData[:,1], c=self.y)
        plt.subplot(1,2,2)
        plt.title("LDA")
        plt.scatter(ldaData[:,0], ldaData[:,1],c=self.y)
        plt.show()
        
    def varVis(self):
        taille_colonne = len(self.features) if len(self.features) < 5 else 5
        taille_ligne = len(self.features) if len(self.features) < 5 else 5
        m = 0
        plt.figure()
        for ligne in range(0, taille_ligne):
            for colonne in range(0, taille_colonne):
                if ligne != colonne:                    
                    m = m + 1
                    plt.subplot(taille_ligne, taille_colonne - 1, m)
                    if self.color is not None:
                        plt.scatter(x=self.x[:,colonne], y=self.x[:,ligne], c=self.color[self.y], alpha=0.8)
                    else:
                        plt.scatter(x=self.x[:,colonne], y=self.x[:,ligne], c=self.y, alpha=0.8)
                    plt.xlabel(self.features[colonne])
                    plt.ylabel(self.features[ligne])
        plt.show()

    def regression_lin(self):
        taille_colonne = len(self.features) if len(self.features) < 5 else 5
        taille_ligne = len(self.features) if len(self.features) < 5 else 5
        m = 0
        plt.figure()
        for ligne in range(0, taille_ligne):
            for colonne in range(0, taille_colonne):
                if ligne != colonne:
                    m = m + 1
                    plt.subplot(taille_ligne, taille_colonne - 1, m)
                    if self.color is not None:
                        plt.scatter(x=self.x[:,colonne], y=self.x[:,ligne], c=self.color[self.y], alpha=0.8)
                    else:
                        plt.scatter(x=self.x[:,colonne], y=self.x[:,ligne], c=self.y, alpha=0.8)
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
    
    
    def classifications(self):
        return [self.classificationModel('KNN'),
        self.classificationModel('CBN'),
        self.classificationModel('MLP'),
        self.classificationModel('SVC')] 

    def classificationModel(self, model='KNN'):
        if model == 'KNN':
            print("Calcul avec l'algorithme du plus proche voisin")
            clf = KNeighborsClassifier()
        elif model == 'CBN':
            print("Calcul avec le classifieur Bayesien Naif")
            clf = GaussianNB()
        elif model == 'MLP':
            print("Calcul avec le Perceptron multi-couche")
            warnings.simplefilter("ignore")
            clf = MLPClassifier()
        elif model == 'SVC':
            print("Calcul avec des machines a vecteurs de support")
            clf = SVC(gamma='auto')
        else :
            return
        tab = []
        for index, data  in enumerate(self.x):
            clf.fit(np.delete(self.x, index, axis=0), np.delete(self.y, index, axis=0))
            tab.append(clf.predict([data])[0])
        return tab, clf.score(self.x, self.y), classification_report(self.y, tab, target_names=self.target_names.astype(str))

    def clusterings(self):
        return [self.clusteringModel('MS'), self.clusteringModel('AP'),
                self.clusteringModel('SC'), self.clusteringModel('DB')]
    
    def clusteringModel(self, model='MS', k=3):
        if model == 'MS':
            clf = MeanShift().fit(self.x)
            return clf.labels_, self.makeScoring(clf.labels_)
        elif model == 'AP':
            clf = AffinityPropagation().fit(self.x)
            return clf.labels_, self.makeScoring(clf.labels_)
        elif model == 'SC':
            clf = SpectralClustering().fit(self.x)
            return clf.labels_, self.makeScoring(clf.labels_)
        elif model == 'DB':
            clf = DBSCAN()
            return clf.fit_predict(self.x), self.makeScoring(clf.fit_predict(self.x))
        else :
            return

    def makeScoring(self, y):
        warnings.simplefilter("ignore")
        a = davies_bouldin_score(self.x, y)
        b = silhouette_score(self.x, y)
        c = adjusted_mutual_info_score(self.y, y)
        d = adjusted_rand_score(self.y, y)

        return np.array([a, b, c, d])
a = Analyse(iris)
b = Analyse(wine)
c = Analyse(digits)

"""
from sklearn.metrics import davies_bouldin_score (X, label)
from sklearn.metrics import silhouette_score (X, label)
from sklearn.metrics import adjusted_mutual_info_score (labels_true, labels_pred, average_method='warn')
from sklearn.metrics import adjusted_rand_score  (labels_true, labels_pred)
"""
