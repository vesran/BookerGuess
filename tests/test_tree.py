from tools.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


def test_gini_and_info_gain():
    filename = 'data/tennis.dat'
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append(line.split())
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    dtree = DecisionTreeClassifier(max_depth=1000, label_names=['Outlook', 'Temp', 'Humidity', 'Wind'])
    dtree.X = X
    dtree.y = y

    assert dtree._gini(0, 'Sunny') == 0.48
    assert dtree._gini(0, 'Overcast') == 0
    assert dtree._gini(0, 'Rain') == 0.48

    assert round(dtree._info_gain(0), 2) == -0.34
    assert round(dtree._info_gain(1), 2) == -0.44
    assert round(dtree._info_gain(2), 2) == -0.37
    assert round(dtree._info_gain(3), 2) == -0.43


def test_decision_tree():
    df = pd.read_csv('./data/titanic.csv', sep=',').drop(['Fare', 'Name', 'Age'], axis=1)
    y = df['Survived'].values
    X = df.drop('Survived', axis=1).values
    dtree = DecisionTreeClassifier(max_depth=10)
    dtree.fit(X, y)

    assert dtree._single_predict(X[0]) == y[0]
    assert dtree._single_predict(X[1]) == y[1]
    assert dtree._single_predict(X[20]) == y[20]