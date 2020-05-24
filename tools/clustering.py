import numpy as np
import random as rnd


def kmeans(X, k, init='randcentroids', eps=1e-10):
    ''' Lloyd's or KMeans algorithm.
    :param X: data
    :param k: number of clusters
    :param init: type of initialisation of centroids/clusters, can be "randcentroids", "randclusters", "kmeans++"
    :param eps: threshold for convergence
    '''

    def dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5

    def dist2(x, centroids):
        return min([dist(x, c) for c in centroids])

    if init == 'randcentroids':
        # Init random centroids by selecting k random example in the given dataset
        numbers = list(range(0, X.shape[0]))
        rnd.shuffle(numbers)
        centroids = [X[numbers[i]] for i in range(k)]
        pred = [0 for x in X]

    elif init == 'randclusters':
        # Attribute a random cluster to each point and compute the centroids
        pred = [rnd.randint(0, k - 1) for _ in X]
        centroids = [np.mean(np.array([X[j] for j in range(X.shape[0]) if pred[j] == i]), axis=0) for i in range(k)]

    elif init == 'kmeans++':
        centroids = [X[rnd.randint(0, X.shape[0] - 1)].tolist()]
        for i in range(1, k):
            # Find the furthest point from the centroids
            furthest_x = X[0]
            for x in X:
                if x.tolist() not in centroids and dist2(x, centroids) > dist2(furthest_x, centroids):
                    furthest_x = x
            centroids.append(furthest_x.tolist())
        pred = [0 for x in X]

    else:
        print('Wrong parameter, init parameter has not been initialized (randcentroids, randclusters, kmeans++).')

    convergence = eps + 1
    nb_iter = 0
    while (convergence >= eps):
        nb_iter += 1
        old_centroids = [c for c in centroids]

        # Update clusters
        for i in range(X.shape[0]):
            # Find closest centroids
            closest_i = 0
            for j in range(len(centroids)):
                if dist(centroids[j], X[i]) < dist(X[i], centroids[closest_i]):
                    closest_i = j
            pred[i] = closest_i

        # Update centroids
        for i in range(len(centroids)):
            centroids[i] = np.mean(np.array([X[j] for j in range(len(X)) if pred[j] == i]), axis=0)

        # Compute convergence value to know if we should stop
        centroids_comparaisons = np.array([dist(old_c, c) for old_c, c in zip(old_centroids, centroids)])
        convergence = np.max(centroids_comparaisons)

    return np.array(pred)