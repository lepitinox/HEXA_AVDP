import pandas as pd
import plotly.express as px
import numpy as np
from sklearn import datasets

class KNearestNeighbors:
    """ K Nearest Neighbors Classifier"""

    def __init__(self, _k):
        """ Initialize the classifier with k value"""
        self.k = _k
        self.X = None
        self.y = None

    def fit(self, _X, _y):
        """ Fit the classifier with training data"""
        self.X = _X
        self.y = _y

    def predict(self, test_X):
        """ Predict the class of the given data"""
        if self.X is None or self.y is None:
            raise Exception('You must fit the data first')
        y_pred = []
        for x in test_X:
            distances = []
            for x_train, y_train in zip(self.X, self.y):
                distance = ((x_train - x) ** 2).sum()
                distances.append([distance, y_train])
            distances = sorted(distances)
            k_nearest_neighbors = distances[:self.k]
            y_pred.append(pd.Series([y for _, y in k_nearest_neighbors]).value_counts().index[0])
        return y_pred

    def score(self, _X, _y):
        """ Return the accuracy of the classifier """
        y_pred = self.predict(_X)
        return sum(y_pred == _y) / len(_y)

    def score_for_non_numerical_y(self, X, y):
        """ Return the accuracy of the classifier for non numerical y"""
        # confusion matrix
        y_pred = self.predict(X)
        confusion_matrix = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'])
        return confusion_matrix


class MinMaxScaler:
    """ Basic Min Max Scaler"""

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def train_test_split(X, y, test_size=0.2, random_state=42):
    """ Split the data into train and test """
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


if __name__ == '__main__':
    # df = pd.read_csv('Iris.csv')
    # X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm']].values
    # y = df['Species'].values

    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Data prep
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    data = {}
    knn = KNearestNeighbors(3)
    knn.fit(X_train, y_train)
    confmatrix = knn.score_for_non_numerical_y(X_test, y_test)
    print(np.diag(confmatrix).sum() / confmatrix.sum().sum())
