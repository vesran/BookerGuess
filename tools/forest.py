from tools.decision_tree import DecisionTreeClassifier, decision_tree2ete
from collections import Counter
import numpy as np
import random as rand


class RandomForestClassifier:

    def __init__(self, n_estimators=1000, max_depth=1000, features_names=None):
        self.n_estimators=n_estimators
        self.max_depth = max_depth
        self.n_features = None
        self.trees = [DecisionTreeClassifier(max_depth=self.max_depth, random_features=True, label_names=features_names)
                      for _ in range(self.n_estimators)]
        self.X, self.y = None, None

    def fit(self, X, y):
        for dtree in self.trees:
            # Bootstrap
            X_train = []
            y_train = []
            for j in range(X.shape[0]):
                i = rand.randint(0, X.shape[0]-1)*0 + j
                X_train.append(X[i])
                y_train.append(y[i])
            X_train, y_train = np.array(X_train), np.array(y_train)
            assert X_train.dtype == X.dtype
            assert X_train.shape[0] == X.shape[0]

            # Training the tree
            dtree.fit(X_train, y_train)

    def _single_predict(self, x):
        # votes = []
        # for i, dtree in enumerate(self.trees):
        #     print(f"{i}e tree")
        #     votes.append(dtree._single_predict(x))

        votes = [dtree._single_predict(x) for dtree in self.trees]
        print(votes)
        return Counter(votes).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._single_predict(x) for x in X])


if __name__ == '__main__':
    import pandas as pd

    # df = pd.read_csv('./resources/titanic.csv', sep=',').drop(['Fare', 'Name', 'Age'], axis=1)
    # y = df['Survived'].values
    # X = df.drop('Survived', axis=1).values
    # rf = RandomForestClassifier(max_depth=10, n_estimators=100)
    #
    # rf.fit(X, y)

    n = 1000
    X = np.array([[rand.choice([1, 2, 3]) for _ in range(20)] for _ in range(n)])
    y = np.array([rand.choice(['a', 'b', 'c']) for _ in range(n)])
    rf = RandomForestClassifier(max_depth=10, n_estimators=10)
    rf.fit(X[:900], y[:900])

    rf._single_predict(X[903])
