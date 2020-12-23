# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from sklearn.decomposition import PCA


def randomInitialCent(K, dataset):
    """
    Compute random initial centroids

    :param K: Number of random centroids
    :param dataset: Data set over which the random centroids will be picked
    :return: Nested list with random points in feature-dimension
    """
    examples = len(dataset) - 1
    random_index = np.random.choice(examples, K)
    randomPoints = []
    for index in list(random_index):
        randomPoints.append(dataset.iloc[index].values.tolist())
    return randomPoints


def get_distance(data_point, centroid):
    """
    Given a data point and a centroid, compute the euclidean distance between both p-dimensional values
    (both arrays have to be of the same length)

    :param data_point: (np.array) Individual point of the data set
    :param centroid: (np.array) Centroid value
    :return: distance as an np.array
    """

    dist = np.sqrt(np.sum((data_point - centroid) ** 2))
    return dist


def mapper(data_point, centroids):
    """
    Given a data point (one out of all examples) find the centroid that is closest to it.
    Return a tuple with the closest centroid and the data point

    :param data_point: (np.array) Individual point of the data set
    :param centroids: (list) Nested list with all centroid values
    :return: Tuple containing as first element the centroid label and as second value the data point provided as input
    """
    distances = [get_distance(data_point, np.array(centroid)) for centroid in centroids]
    centroid_label = np.argmin(distances)
    yield (centroid_label, data_point)


def reducer(centroid_label, data_points):
    """
    Given a centroid label and all the data_points associated to this centroid label compute the new center

    :param centroid_label: Cluster label and all the data points belonging to this cluster
    :param data_points: All data points that belong to the provided centroid label
    :return: Tuple with first position the same centroid label and second the average of the data points
    """

    yield (centroid_label, np.average(data_points, axis=0))


def apply_mapper(dataset, centroids):
    """
    Given a data set and some centroids, apply the mapper over all data points in the data set (dataset).
    The output will provide the data points closest to the centroids

    :param dataset: Data set over which we will apply the mapper function row by row
    :param centroids: Potential centroids
    :return: All data points closest to each centroid label (cluster group)
    """
    collector = defaultdict(list)

    for data_points in dataset.values.tolist():
        for centroid_label, data_point in mapper(data_points, centroids):
            collector[centroid_label].append(data_point)

    return collector


def apply_reducer(collector):
    """
    Apply the reduce function, i.e. the output of this function, will return the re-centered centroids
    """
    return [output for cluster_label, data_points in collector.items()
            for output in reducer(cluster_label, data_points)]


def centroid_diff(old_centroids, new_centroids):
    """
    Return the distance (euclidean) between two centroids.
    This function will be used to break the k-means iteration
    """

    return np.sqrt(np.sum((np.array(old_centroids) - np.array(new_centroids)) ** 2))


def k_means_map_reduce(X, k, max_iter=100, tolerance=0.00001):
    """
    Given a data set (X) and a number of cluster, apply k-means using a mapreduce approach

    :param X: Data set including only the features over which the k-means will be done
    :param k: number of clusters
    :param max_iter: Iterations until we stop loop
    :param tolerance: Difference between centroid tolerance
    :return: data frame with example points and clusters
    """
    # Initialize random example points from data set
    centroids = randomInitialCent(k, X)
    count = 0
    while True:
        count += 1
        # 1. Apply the mapper over the data set and the initial centroid values
        collector = apply_mapper(X, centroids)
        # 2. Store centroids for later use
        old_centroids = centroids

        # 3. Plot data points that are assigned to each cluster respectively
        plt.figure(figsize=(20, 10))

        df_temp = pd.DataFrame([])
        clusters = []
        for i in collector:
            temp_list = collector[i]
            df_temp = df_temp.append(temp_list, ignore_index=True)
            list_cluster = [i] * len(temp_list)
            clusters = clusters + list_cluster

        # 4. For plotting purposes we will reduce the dimensionality of the data to two components
        pca = PCA(n_components=2)
        scatter_plot_points = pca.fit_transform(df_temp)
        centroids_pca = pca.fit_transform(centroids)

        plt.figure(figsize=(20, 10))
        sns.scatterplot(x=scatter_plot_points[:, 0],
                        y=scatter_plot_points[:, 1],
                        hue=clusters)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=130, marker="x", color='black', linewidths=3)

        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Iteration ' + str(count))
        plt.show();

        # 5. Given the new cluster groups, reajust the centroids using the reduce functionality
        centroids = apply_reducer(collector)
        centroids = [list(centroids[i][1]) for i in range(len(centroids))]

        # 6. Check whether the old centroids and the newly computed centroids are far from each other (using the
        # threshold value)
        diff = centroid_diff(old_centroids, centroids)

        if (diff <= tolerance) | (count > max_iter):
            column_names = ['feature_' + str(i) for i in range(1, len(df_temp.columns) + 1)]
            df_temp.columns = column_names
            df_temp['Cluster'] = clusters
            return df_temp
