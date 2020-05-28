from tools.decision_tree import DecisionTreeClassifier
from collections import Counter
import numpy as np
import random as rand


class RandomForestClassifier:

    def __init__(self, n_estimators=1000, max_depth=1000, features_names=None, sampfactor=1.0, class_weights=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_features = None
        self.trees = [DecisionTreeClassifier(max_depth=self.max_depth, random_features=True, label_names=features_names)
                      for _ in range(self.n_estimators)]
        self.X, self.y = None, None
        self.sampfactor = sampfactor
        self.classes_ = None
        self.class_weights = class_weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        sampsize = int(X.shape[0] * self.sampfactor)
        assert sampsize > 0
        for dtree in self.trees:
            # Bootstrap
            X_train = []
            y_train = []
            for j in range(sampsize):
                i = rand.randint(0, X.shape[0]-1)
                X_train.append(X[i])
                y_train.append(y[i])
            X_train, y_train = np.array(X_train), np.array(y_train)
            assert X_train.dtype == X.dtype

            # Handle class weights
            if self.class_weights == 'balanced_subsample':
                weights = {label: 1-(y_train.tolist().count(label)/y_train.shape[0]) for label in np.unique(y_train)}
            else:
                weights = self.class_weights
            dtree.class_weights = weights

            # Training the tree
            dtree.fit(X_train, y_train)

    def _single_predict(self, x, debug=False):
        votes = [dtree._single_predict(x) for dtree in self.trees]
        print(f"Votes : {votes}") if debug else 0
        return Counter(votes).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._single_predict(x) for x in X])

    def _predict_single_proba(self, x):
        preds = [dtree._single_predict(x) for dtree in self.trees]
        res = []
        for label in self.classes_:
            occ = preds.count(label)
            res.append(occ)
        return np.array(res) / self.n_estimators

    def predict_proba(self, X):
        return np.array([self._predict_single_proba(x) for x in X])


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('./resources/titanic.csv', sep=',').drop(['Fare', 'Name', 'Age'], axis=1)
    y = df['Survived'].values
    X = df.drop('Survived', axis=1).values
    rf = RandomForestClassifier(max_depth=10, n_estimators=100)

    rf.fit(X[:800], y[:800])
    (rf.predict(X[800:]) == y[800:]).astype(int).sum() / X[800:].shape[0]
