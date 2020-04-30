# TODO: confusion matrix
# TODO: cross validation

import numpy as np

# Compare with sklearn metrics
from sklearn.model_selection import cross_val_score


def confusion_matrix(classifier, X, y, normalize=True, decimal=2):
    labels = np.unique(y)
    y_pred = classifier.predict(X)
    counts = np.zeros((labels.shape[0], labels.shape[0]))
    for prediction, truth in zip(y_pred, y):
        i_truth = np.where(labels == truth)[0][0]
        i_prediction = np.where(labels == prediction)[0][0]
        counts[i_truth][i_prediction] += 1

    if normalize:
        counts = np.round(counts / np.sum(counts, axis=1).reshape(counts.shape[0], 1), decimal)
    return counts


if __name__ == '__main__':
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

    confusion_matrix(classifier, X_test, y_test, normalize=True)