from tools.forest import RandomForestClassifier
import pandas as pd
import numpy as np
import random as rand


def test_titanic():
    df = pd.read_csv('./resources/titanic.csv', sep=',').drop(['Fare', 'Name', 'Age'], axis=1)
    y = df['Survived'].values
    X = df.drop('Survived', axis=1).values
    rf = RandomForestClassifier(max_depth=10, n_estimators=100, sampfactor=0.6)

    rf.fit(X[:800], y[:800])
    rf_score = (rf.predict(X[800:]) == y[800:]).astype(int).sum() / X[800:].shape[0]
    assert rf_score > 0.8


def test_learning():
    n = 500
    n_features = 20
    X = np.array([[rand.choice([1, 2, 3]) for _ in range(n_features)] for _ in range(n)])
    y = np.array([rand.choice(['a', 'b', 'c']) for _ in range(n)])
    rf = RandomForestClassifier(max_depth=8, n_estimators=20, sampfactor=0.6)
    rf.fit(X, y)
    prediction = rf._single_predict(X[6], debug=True)
    assert prediction == y[6]

