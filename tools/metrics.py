import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(classifier, X, y, normalize=True, decimal=2, plot=False):
    labels = np.unique(y)
    y_pred = classifier.predict(X)
    counts = np.zeros((labels.shape[0], labels.shape[0]))
    for prediction, truth in zip(y_pred, y):
        i_truth = np.where(labels == truth)[0][0]
        i_prediction = np.where(labels == prediction)[0][0]
        counts[i_truth][i_prediction] += 1

    # Normalize
    if normalize:
        counts = np.round(counts / np.sum(counts, axis=1).reshape(counts.shape[0], 1), decimal)

    if plot:
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.matshow(counts, cmap=plt.cm.Blues)
        ax.set_ylabel('Truth')
        ax.set_xlabel('Prediction')
        ax.set_xticklabels([0] + labels.tolist())
        ax.set_yticklabels([0] + labels.tolist())
        for (i, j), z in np.ndenumerate(counts):
            color = 'black' if counts[i][j] <= 0.5 or not normalize else 'white'
            ax.text(j, i, f'{{:0.{decimal}f}}'.format(z), ha='center', va='center', c=color)
    return counts


def recall(bi_confmat):
    assert bi_confmat.shape == (2, 2)
    return bi_confmat[0][0] / (bi_confmat[0][0] + bi_confmat[0][1])


def precision(bi_confmat):
    assert bi_confmat.shape == (2, 2)
    return bi_confmat[0][0] / (bi_confmat[0][0] + bi_confmat[1][0])


def f1_score(y_pred, y_true):
    labels = np.unique(y_true)
    counts = np.zeros((labels.shape[0], labels.shape[0]))
    for prediction, truth in zip(y_pred, y_true):
        i_truth = np.where(labels == truth)[0][0]
        i_prediction = np.where(labels == prediction)[0][0]
        counts[i_truth][i_prediction] += 1
    prec = precision(counts)
    rec = recall(counts)
    return 2 * rec * prec / (rec + prec)


def score(y_pred, y_true):
    return np.sum((y_pred == y_true).astype(int)) / y_true.shape[0]


def cross_validation_score(model, X, y, k=3, scorer=score):
    n = X.shape[0]
    p = np.random.permutation(len(X))
    X, y = X[p], y[p]
    partition_X = [X[int((i-1)*n/k):int(i*n/k)] for i in range(1, k+1)]
    partition_y = [y[int((i-1)*n/k):int(i*n/k)] for i in range(1, k+1)]
    scores = []

    for i in range(len(partition_X)):
        # Get train data
        train_X = partition_X[:]
        train_X.pop(i)
        train_X = np.concatenate(train_X)
        train_y = partition_y[:]
        train_y.pop(i)
        train_y = np.concatenate(train_y)

        # Train and score
        model.fit(train_X, train_y)  # Suppose fit makes the model forget previously learnt data
        y_pred = model.predict(partition_X[i])
        scores.append(scorer(y_pred, partition_y[i]))
    return np.array(scores)


if __name__ == '__main__':
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    model = DecisionTreeClassifier()
    cross_val_score(model, X, y).mean()

    cross_validation_score(model, X, y, k=10, scorer=f1_score).mean()
