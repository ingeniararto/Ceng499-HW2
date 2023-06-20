import pickle
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from Part1.Knn import KNN
from Part1.Distance import Distance


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))

distance = Distance.calculateCosineDistance

kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

k_values = [1, 5, 10, 15, 19]
distances = [Distance.calculateCosineDistance, Distance.calculateMahalanobisDistance, Distance.calculateMinkowskiDistance]

"""
First configuration cosine similarity
"""

for K in k_values:
    print(f"Configuration: k={K}, cosine")
    accuracies = np.zeros(shape=50)
    i = 0
    for train_indices, test_indices in kfold.split(dataset, labels):
        current_train = dataset[train_indices]
        current_train_label = labels[train_indices]
        knn = KNN(current_train, current_train_label, Distance.calculateCosineDistance, K=K)

        X_test = dataset[test_indices]
        y_test = labels[test_indices]

        accuracy = knn.calculate_accuracy(test_data=X_test, test_label=y_test)

        accuracies[i] = accuracy
        i += 1
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    conf_int_max = avg_accuracy + 1.96*(std_accuracy/np.sqrt(50))
    conf_int_low = avg_accuracy - 1.96*(std_accuracy/np.sqrt(50))
    print(f"Average Accuracy %{avg_accuracy*100}")
    print(f"Confidence Interval [{conf_int_low},{conf_int_max}]")

"""
Second configuration Minkowski distance
"""
print("*****************************")
for K in k_values:
    print(f"Configuration: k={K}, Minkovski")
    accuracies = np.zeros(shape=50)
    i=0
    for train_indices, test_indices in kfold.split(dataset, labels):
        current_train = dataset[train_indices]
        current_train_label = labels[train_indices]
        knn = KNN(current_train, current_train_label, Distance.calculateMinkowskiDistance, similarity_function_parameters=2, K=K)

        X_test = dataset[test_indices]
        y_test = labels[test_indices]

        accuracy = knn.calculate_accuracy(test_data=X_test, test_label=y_test)

        accuracies[i] = accuracy
        i += 1
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    conf_int_max = avg_accuracy + 1.96*(std_accuracy/np.sqrt(50))
    conf_int_low = avg_accuracy - 1.96*(std_accuracy/np.sqrt(50))
    print(f"Average Accuracy %{avg_accuracy*100}")
    print(f"Confidence Interval [{conf_int_low},{conf_int_max}]")

"""
Third configuration Mahalanobis distance
"""
print("*****************************")
for K in k_values:
    print(f"Configuration: k={K}, Mahalanobis")
    i = 0
    accuracies = np.zeros(shape=50)
    for train_indices, test_indices in kfold.split(dataset, labels):
        current_train = dataset[train_indices]
        current_train_label = labels[train_indices]
        s_minus_1 = np.linalg.inv(np.cov(current_train, rowvar=False))
        knn = KNN(current_train, current_train_label, Distance.calculateMahalanobisDistance, similarity_function_parameters=s_minus_1, K=K)
        X_test = dataset[test_indices]
        y_test = labels[test_indices]

        accuracy = knn.calculate_accuracy(test_data=X_test, test_label=y_test)

        accuracies[i] = accuracy
        i += 1
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    conf_int_max = avg_accuracy + 1.96*(std_accuracy/np.sqrt(50))
    conf_int_low = avg_accuracy - 1.96*(std_accuracy/np.sqrt(50))
    print(f"Average Accuracy %{avg_accuracy*100}")
    print(f"Confidence Interval [{conf_int_low},{conf_int_max}]")
