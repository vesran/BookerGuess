from tools.metrics import confusion_matrix, recall, precision, f1score


class Model3:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return ['a', 'b', 'c', 'b', 'a', 'c']


class Model2:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return ['a', 'a', 'a', 'b', 'b', 'b']


def test_confusion_matrix():
    model = Model3()
    X, y = [], ['b', 'b', 'c', 'b', 'a', 'c']
    m = confusion_matrix(model, X, y, plot=False, decimal=1)
    assert sum(m[0]) == 1
    assert sum(m[1]) == 1
    assert sum(m[2]) == 1


def test_recall():
    model = Model2()
    y = ['a', 'a', 'b', 'a', 'b', 'b']
    m = confusion_matrix(model, [], y, plot=False, decimal=1, normalize=False)
    rec = recall(m)
    assert rec == 2/3


def test_precision():
    model = Model2()
    y = ['a', 'a', 'b', 'a', 'b', 'b']
    m = confusion_matrix(model, [], y, plot=False, decimal=1, normalize=False)
    prec = precision(m)
    assert prec == 2/3


def test_f1score():
    model = Model2()
    y = ['a', 'a', 'b', 'a', 'b', 'b']
    m = confusion_matrix(model, [], y, plot=False, decimal=1, normalize=False)
    f1 = f1score(m)
    assert f1 == 2/3
