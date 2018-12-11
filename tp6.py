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
        self.features = data.feature_names
        self.color = np.array(['royalblue', 'g', 'darkorange'])
        self.target_names = data.target_names if data.target_names is not None else None

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
    
    
    def classifications(self):
        return [self.classificationModel('KNN'),
        self.classificationModel('CBN'),
        self.classificationModel('MLP'),
        self.classificationModel('SVC')] 

    def classificationModel(self, model='KNN'):
        if model == 'KNN':
            clf = KNeighborsClassifier()
        elif model == 'CBN':
            clf = GaussianNB()
        elif model == 'MLP':
            warnings.simplefilter("ignore")
            clf = MLPClassifier()
        elif model == 'SVC':
            clf = SVC(gamma='auto')
        else :
            return
        tab = []
        for index, data  in enumerate(self.x):
            clf.fit(np.delete(self.x, index, axis=0), np.delete(self.y, index, axis=0))
            tab.append(clf.predict([data])[0])
        return tab, clf.score(self.x, self.y), classification_report(self.y, tab, target_names=self.target_names)

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
from sklearn.metrics import adjusted_mutual_info_score (labels_true, labels_pred, average_method=’warn’)
from sklearn.metrics import adjusted_rand_score  (labels_true, labels_pred)
"""








