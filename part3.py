import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

"""plot_dendogram function code is taken from scikit-learn: 
https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html.
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
"""
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    sc.dendrogram(linkage_matrix, **kwargs)


linkages = ['single', 'complete']
distances = ['euclidean', 'cosine']
k_values = [2, 3, 4, 5]


for linkage in linkages:
    for distance in distances:
        silhouette_scores = []
        for K in k_values:
            hac = AgglomerativeClustering(n_clusters=K, affinity=distance, linkage=linkage,
                                          compute_distances=True)
            hac.fit(dataset)
            s = silhouette_score(dataset, hac.labels_, metric=distance)
            silhouette_scores.append(s)
            plt.title("Hierarchical Agglomerative Clustering Dendrogram")
            plot_dendrogram(hac)
            plt.xlabel(f"linkage:{linkage}, distance:{distance}, cluster_size:{K}")
            plt.show()

        print(silhouette_scores)
        plt.plot(k_values, silhouette_scores)
        plt.title(f'Silhouette Scores vs K values for {distance}-{linkage}') # Title of the plot
        plt.xlabel('K values') # X-Label
        plt.ylabel('Silhouette scores') # Y-Label
        plt.show()