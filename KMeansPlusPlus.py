import numpy as np

class KMeansPlusPlus:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: np.zeros(shape=(0, len(dataset[0]))) for i in range(K)}

        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: np.random.rand(len(dataset[0])) for i in range(K)}

        # you are free to add further variables and functions to the class
        # dictionary for which cluster every data instance is assigned
        self.dataset_cluster_dict = {i: 0 for i in range(len(dataset))}

    def euclidean_distance(self, x, y):
        return np.linalg.norm(x-y)

    def init_centers(self):
        n_data = len(self.dataset)
        indexes = []
        index = np.random.random_integers(0, n_data-1)
        indexes.append(index)
        self.cluster_centers[0] = self.dataset[index]
        for i in range(1, self.K):
            distances = np.zeros(shape=n_data)
            for n in range(n_data):
                if n not in indexes:
                    min_distance = 9999
                    for j in range(i):
                        distance = self.euclidean_distance(self.cluster_centers[j], self.dataset[n])
                        if distance < min_distance:
                            min_distance = distance
                    distances[n] = min_distance
                else:
                    distances[n] = 0
            distances = np.power(distances, 2)

            sum = np.sum(distances)
            distances = np.divide(distances, sum)
            index = np.random.choice(np.arange(0, n_data), p=distances)
            indexes.append(index)
            self.cluster_centers[i] = self.dataset[index]

    def calculate_difference(self, x, y):
        return np.power(self.euclidean_distance(x, y), 2)

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        sum = 0
        for i in range(len(self.dataset)):
            center_index = self.dataset_cluster_dict[i]
            sum += self.calculate_difference(self.cluster_centers[center_index], np.array(self.dataset[i]))
        return sum

    def update_center(self):
        for i in range(self.K):
            new_center = np.mean(self.clusters[i], axis=0)
            comparison = self.cluster_centers[i] == new_center
            self.cluster_centers[i] = new_center

    def run(self):
        """Kmeans algorithm implementation"""
        self.init_centers()
        loss = 99999999
        while loss > self.calculateLoss():
            self.clusters = {i: np.zeros(shape=(0, len(self.dataset[0]))) for i in range(self.K)}
            for n in range(len(self.dataset)):
                min_distance = 99999999
                index = 0
                for i in range(self.K):
                    distance = self.euclidean_distance(self.dataset[n], self.cluster_centers[i])
                    if distance < min_distance:
                        min_distance = distance
                        index = i

                self.clusters[index] = np.append(self.clusters[index], [self.dataset[n]], axis=0)
                self.dataset_cluster_dict[n] = index

            self.update_center()
            loss = self.calculateLoss()

        return self.cluster_centers, self.clusters, loss


