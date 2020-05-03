# TODO : Decision tree
# TODO : Random forest

import numpy as np
from collections import Counter


class DecisionTreeClassifier:
    """ Decision Tree for categorical variables and classification using CART """

    def __init__(self, max_depth, label_names=None, _i_split=None, _i_var=None, _value=None):
        self.max_depth = max_depth
        self.X, self.y = None, None
        self.children = None
        self.label_names = label_names
        self._i_split = _i_split
        self._i_var = _i_var
        self._value = _value

    def fit(self, X, y):
        self.X, self.y = X, y
        self.label_names = list(range(self.X.shape[1])) if self.label_names is None else self.label_names
        to_visit = [self]
        i = 0
        while len(to_visit) > 0:
            print(f"Iteration {i}")
            i += 1
            to_develop = to_visit.pop(0)
            to_develop.children = to_develop._split()
            print("Number of children :", len(to_develop.children))
            to_visit.extend([child for child in to_develop.children if child._is_splitable()])
            print(f"to_visit : {to_visit}")

    def _is_splitable(self):
        return self.max_depth > 0 and np.unique(self.y).shape[0] > 1 and self.X.shape[1] > 1

    def _gini(self, i_var, value):
        score = 0
        positions = np.where(self.X[:, i_var] == value)[0]
        sub_y = self.y[positions]
        for label in np.unique(self.y):
            nb_label = np.where(sub_y == label)[0].shape[0] / sub_y.shape[0]
            score += (nb_label ** 2)
        return 1 - score

    def _info_gain(self, i_var):
        score = 0
        for value in np.unique(self.X[:, i_var]):
            proportion = np.where(self.X[:, i_var] == value)[0].shape[0] / self.X.shape[0]
            gini_score = self._gini(i_var, value)
            score += (proportion * gini_score)
        return -score

    def _split(self):
        gini_scores = []
        for i_var in range(self.X.shape[1]):
            score = self._info_gain(i_var)
            gini_scores.append(score)
        print(gini_scores)
        self._i_split = np.argmax(gini_scores)
        print(f"Split on var{self._i_split}")

        children = []
        for value in np.unique(self.X[:, self._i_split]):
            names = None
            if self.label_names is not None:
                names = self.label_names[:]
                names.pop(self._i_split)
            child = DecisionTreeClassifier(max_depth=self.max_depth-1, _i_var=self._i_split, _value=value, label_names=names)
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
            value = x[current._i_split]  # Value to consider
            for child in current.children:
                if child._value == value:
                    x = np.delete(x, current._i_split)
                    current = child
                    break
        return Counter(current.y).most_common(1)[0][0]


def decision_tree2ete(dtree):
    """Converts a decision tree to an ete3 tree for visualisation"""
    res_tree = ete3.Tree()
    res_tree.name = dtree.describe()
    dtree_to_visit = [dtree]
    ete_to_visit = [res_tree]

    while len(dtree_to_visit) > 0:
        current_dtree = dtree_to_visit.pop(0)
        current_ete = ete_to_visit.pop(0)

        for dchild in current_dtree.children:
            ete_child = ete3.Tree(name=dchild.describe())
            current_ete.add_child(ete_child)

            # Visit child's children if it's not a leaf
            if dchild.children is not None:
                dtree_to_visit.append(dchild)
                ete_to_visit.append(ete_child)

    return res_tree


if __name__ == '__main__':
    from ete3 import AttrFace, faces
    import ete3
    import pandas as pd

    df = pd.read_csv('./data/titanic.csv', sep=',').drop(['Fare', 'Name', 'Age'], axis=1)
    y = df['Survived'].values
    X = df.drop('Survived', axis=1).values
    dtree = DecisionTreeClassifier(max_depth=10, label_names=['Pclass', 'Sex', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'])
    dtree.fit(X, y)
    # print(dtree._single_predict(X[0]))

    def my_layout(node):
        # Adds the name face to the image at the preferred position
        name_face = AttrFace("name")
        faces.add_face_to_node(name_face, node, column=0, position="branch-right")


    # Tree style parameters
    ts = ete3.TreeStyle()
    ts.scale = 100  # 100 pixels per branch length unit
    ete_tree = decision_tree2ete(dtree)
    ts.show_leaf_name = False
    ts.layout_fn = my_layout

    # Display
    print(ete_tree.get_ascii(show_internal=True))
    # ete_tree.show(tree_style=ts)