import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression

iris = datasets.load_iris()
x = datasets.load_iris().data
y = datasets.load_iris().target

class Analyse:
    def __init__(self, data):
        self.x = data.data
        self.y = data.target
        self.features = data.feature_names
        self.color = np.array(['royalblue', 'g', 'darkorange'])
        self.pd = pd.DataFrame(data=np.column_stack((data.data,data.target_names[data.target])), columns=[ s.replace(' ', '_') for s in data.feature_names]+["label"])

# sns.load_dataset("iris")

    def analyse(self):
        print("Le nombre de données est " + str(self.x.size))
        print("Le nombre de variable est " + str(self.x.shape[1]))
        print("Les numeros de classes sont ")

    def visualisation(self):
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
        """
        fig, axs = plt.subplots(len(self.features), len(self.features))
        for k, a in enumerate(self.features):
            for l, b in enumerate(self.features):
                if k == l :
                    axs[k, l].plot(self.x[:,k], c='y', alpha=0.8)
                    axs[k, l].plot(self.x[:,l], c='r', alpha=0.8)
                else :
                    axs[k, l].scatter(self.x[:,l], self.x[:,k], c=self.color[self.y], alpha=0.8)
        """
        g = sns.pairplot(self.pd, hue="label")
        plt.show()

    def regression_lin(self):
        lm = LinearRegression().fit(self.x, self.y)
        data = lm.predict(self.x)
        plt.figure()
        plt.plot(self.y, data, ' .')
        x = np.linspace(0, 330, 100)
        y = x
        plt.plot(x, y)
        #plt.show()
        
        # markers=["o", "s", "D"]
        
a = Analyse(iris)
a.varVis()
