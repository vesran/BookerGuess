from tools.decision_tree import DecisionTreeClassifier
import numpy as np


class RandomForestClassifier:

    def __init__(self, n_estimators=1000, max_depth=1000):
        self.n_estimators=n_estimators
        self.max_depth = max_depth
        self.n_features = None
        self.trees = [DecisionTreeClassifier(max_depth=self.max_depth) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


if __name__ == '__main__':
    pass