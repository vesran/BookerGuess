import numpy as np
import random as rand
from collections import Counter


class DecisionTreeClassifier:
    """ Decision Tree for categorical variables and classification using CART """

    def __init__(self, max_depth, label_names=None, _i_split=None, _i_var=None, _value=None, random_features=False,
                 class_weights=None):
        self.max_depth = max_depth
        self.X, self.y = None, None
        self.children = None
        self.label_names = label_names
        self._i_split = _i_split
        self._i_var = _i_var
        self._value = _value
        self.random_features = random_features
        self.class_weights = class_weights

    def fit(self, X, y):
        self.X, self.y = X, y
        self.label_names = list(range(self.X.shape[1])) if self.label_names is None else self.label_names
        to_visit = [self]
        while len(to_visit) > 0:
            to_develop = to_visit.pop(0)
            to_develop.children = to_develop._split()  # Children are reset every time fit is called
            to_visit.extend([child for child in to_develop.children if child._is_splitable()])

    def _is_splitable(self):
        return self.max_depth > 0 and np.unique(self.y).shape[0] > 1 and self.X.shape[1] > 1

    def _gini(self, i_var, value):
        self.class_weights = self.class_weights if self.class_weights is not None else {label: 1.0 for label in np.unique(self.y)}
        score = 0
        positions = np.where(self.X[:, i_var] == value)[0]
        sub_y = self.y[positions]
        for label in np.unique(self.y):
            prop_label = self.class_weights[label] * np.where(sub_y == label)[0].shape[0] / sub_y.shape[0]  # Weighted !
            score += (prop_label ** 2)
        return 1 - score

    def _info_gain(self, i_var):
        score = 0
        for value in np.unique(self.X[:, i_var]):
            proportion = np.where(self.X[:, i_var] == value)[0].shape[0] / self.X.shape[0]
            gini_score = self._gini(i_var, value)
            score += (proportion * gini_score)
        return -score

    def _split(self):
        if self.random_features:
            indexes = rand.sample(set(range(self.X.shape[1])), int(np.ceil(self.X.shape[1] ** 0.5)))
            gini_scores = [self._info_gain(i_var) if i_var in indexes else -100 for i_var in range(self.X.shape[1])]
        else:
            gini_scores = [self._info_gain(i_var) for i_var in range(self.X.shape[1])]
        self._i_split = np.argmax(gini_scores)

        children = []
        for value in np.unique(self.X[:, self._i_split]):
            names = None
            if self.label_names is not None:
                names = self.label_names[:]
                names.pop(self._i_split)
            child = DecisionTreeClassifier(max_depth=self.max_depth-1, _i_var=self._i_split, _value=value,
                                           label_names=names, random_features=self.random_features,
                                           class_weights=self.class_weights)
            positions = np.where(self.X[:, self._i_split] == value)[0]
            child.X = np.delete(self.X[positions], self._i_split, axis=1)
            child.y = self.y[positions]
            children.append(child)
        return children

    def describe(self):
        if self.label_names is not None:
            value = self._value
            varname = self.label_names[self._i_split] if self._i_split is not None else self.y[0]
        else:
            value = self._value
            varname = self._i_split
        return f"={value} | Split {varname}"

    def _single_predict(self, x):
        current = self
        while current.children is not None:
            loop = True
            value = x[current._i_split]  # Value to consider
            for child in current.children:
                if child._value == value:
                    x = np.delete(x, current._i_split)
                    current = child
                    loop = False
                    break
            if loop:
                break
        counts = Counter(current.y)
        for k in counts:
            counts[k] *= self.class_weights[k]
        return counts.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._single_predict(x) for x in X])
