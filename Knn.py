import numpy as np


class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def calculate_accuracy(self, test_data, test_label):
        sum = 0
        for i in range(len(test_data)):
            if self.predict(test_data[i]) == test_label[i]:
                sum += 1
        return sum/len(test_data)

    def predict(self, instance):
        nearest_neighbors = dict()
        for i in range(len(self.dataset)):
            if self.similarity_function_parameters is None:
                nearest_neighbors[i] = self.similarity_function(instance, self.dataset[i])
            else:
                nearest_neighbors[i] = self.similarity_function(instance, self.dataset[i],
                                                                self.similarity_function_parameters)
        sorted_nearest_neighbors = [self.dataset_label[k] for k, v in (sorted(nearest_neighbors.items(), key=lambda item: item[1]))[0:self.K]]
        return np.bincount(np.asarray(sorted_nearest_neighbors)).argmax()


