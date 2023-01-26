from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, OPTICS, cluster_optics_dbscan, AgglomerativeClustering
from shapely.geometry import Point
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def perform_DBSCAN(data, radius, min_points, noise, cols):
    subset = data[cols]
    scaled = StandardScaler().fit_transform(subset)

    # perform DBSCAN
    db = DBSCAN(eps=radius, min_samples=min_points).fit(scaled)

    # add cluster labels
    labels = db.labels_
    data["cluster"] = labels

    try:
        print("DBSCAN S_SCORE:", metrics.silhouette_score(scaled, labels, metric='euclidean'))
    except:
        pass
    try:
        print("DBSCAN CH_SCORE:", metrics.calinski_harabasz_score(scaled, labels))
    except:
        pass

    if not noise:
        return data[data["cluster"] != -1]

    return data


import hdbscan


def perform_HDBSCAN(data, noise, cols, r=0.6):
    subset = data[cols]
    scaled = StandardScaler().fit_transform(subset)

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=80,
                                cluster_selection_epsilon=r,
                                cluster_selection_method='eom',
                                gen_min_span_tree=True)
    clusterer.fit(scaled)

    labels = clusterer.labels_
    # add cluster labels
    data["cluster"] = labels

    try:
        print("HDBSCAN S_SCORE:", metrics.silhouette_score(scaled, labels, metric='euclidean'))
    except:
        pass
    try:
        print("HDBSCAN CH_SCORE:", metrics.calinski_harabasz_score(scaled, labels))
    except:
        pass

    # print("HDBSCAN DBCV_SCORE:", DBCV(scaled, labels, dist_function=euclidean))

    if not noise:
        return data[data["cluster"] != -1]

    return data


def get_clusters(data, cols, method, r, mp, noise=False):
    """Calls clustering method and calculates centroids

    Args:
        data (DataFrame): The data to cluster on.
        cols (list): The feature space used to calculate clusters.
        method (str): The clustering method to use (Options include ["DBSCAN", "OPTICS"]).
        r (float, optional): Radius for DBSCAN. Defaults to 0.2.
        mp (int, optional): MinPoints (epsilon) for DBSCAN. Defaults to 50.
        noise (bool, optional): Return points in the noise cluster (-1 label). Defaults to False.

    Returns:
        (clusters, centroids): Tuple containing the clusters and centroids DataFrame and GeoDataFrame, respectively.
    """

    if method == "DBSCAN":
        clusters = perform_DBSCAN(data,
                                  radius=r,
                                  min_points=mp,
                                  noise=noise,
                                  cols=cols
                                  )
    #elif method == "OPTICS":
    #    clusters = perform_OPTICS(data, noise=noise, cols=cols, r=r)

    elif method == "HDBSCAN":
        clusters = perform_HDBSCAN(data, noise=noise, cols=cols, r=r)

    #elif method == "AGGLO":
    #    clusters = perform_AGGLO(data, noise=noise, cols=cols)

    # calculate centroids
    grouped = clusters.groupby("cluster")
    centroids = grouped[cols].apply(np.mean)
    centroids.index.name = "index"
    centroids["cluster"] = centroids.index
    centroids["geometry"] = centroids.apply(lambda row: Point([row["location-long"], row["location-lat"]]), axis=1)

    return clusters, centroids


import seaborn as sns


def plot_range(clusters, centroids, ax=None, show=True):
    """
    plots clusters and centroids for ONE elephant
    """

    # plot clusters
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    sns.set_style("white")
    sns.despine()
    sns.scatterplot(data=clusters,
                    x="location-long",
                    y="location-lat",
                    hue="cluster",
                    palette="Paired",
                    s=4,
                    ax=ax
                    )

    # plot_centroids(centroids, ax, color_legend=False)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Legend")

    if show:
        plt.show()
    else:
        return ax