# TODO: confusion matrix, precision, recall, f1-score
# TODO: cross validation

import numpy as np
import matplotlib.pyplot as plt

# Compare with sklearn metrics
from sklearn.model_selection import cross_val_score


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
            color = 'black' if counts[i][j] <= 0.5 else 'white'
            ax.text(j, i, f'{{:0.{decimal}f}}'.format(z), ha='center', va='center', c=color)
    return counts


def recall(bi_confmat):
    assert bi_confmat.shape == (2, 2)
    return bi_confmat[0][0] / (bi_confmat[0][0] + bi_confmat[0][1])


def precision(bi_confmat):
    assert bi_confmat.shape == (2, 2)
    return bi_confmat[0][0] / (bi_confmat[0][0] + bi_confmat[1][0])


def f1score(bi_confmat):
    prec = precision(bi_confmat)
    rec = recall(bi_confmat)
    return 2 * rec * prec / (rec + prec)


if __name__ == '__main__':
    pass