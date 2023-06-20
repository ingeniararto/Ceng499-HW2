import numpy as np
import matplotlib.pyplot as plt
from Part2.KMeansPlusPlus import KMeansPlusPlus
import pickle


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))


k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

range_val = 10

#
# print("Dataset 1 with Kmeans++")
# averages_1 = np.zeros(shape=len(k_values))
# index = 0
# for K in k_values:
#     min_losses = np.zeros(shape=range_val)
#     for i in range(range_val):
#         losses = np.zeros(shape=range_val)
#         for j in range(range_val):
#             kmeans = KMeansPlusPlus(dataset1, K)
#             cluster_centers, clusters, loss = kmeans.run()
#             losses[j] = loss
#         min_losses[i] = np.min(losses)
#     avg_losses = np.mean(min_losses)
#     std_losses = np.std(min_losses)
#     conf_interval_max = avg_losses + 1.96*(std_losses/np.sqrt(range_val))
#     conf_interval_min = avg_losses - 1.96*(std_losses/np.sqrt(range_val))
#     print(f"K:{K}, Average Loss:{avg_losses}, Confidence Interval:[{conf_interval_min},{conf_interval_max}]")
#     averages_1[index] = avg_losses
#     index += 1
#
# plt.plot(k_values, averages_1)
# plt.title('Dataset 1 with Kmeans++') # Title of the plot
# plt.xlabel('K values') # X-Label
# plt.ylabel('Loss value') # Y-Label
# plt.show()


#############################################################





print("*********************************")
print("Dataset 2 with Kmeans++")
averages_2 = np.zeros(shape=len(k_values))
index = 0
for K in k_values:
    min_losses = np.zeros(shape=range_val)
    for i in range(range_val):
        losses = np.zeros(shape=range_val)
        for j in range(range_val):
            kmeans = KMeansPlusPlus(dataset2, K)
            cluster_centers, clusters, loss = kmeans.run()
            losses[j] = loss
        min_losses[i] = np.min(losses)
    avg_losses = np.mean(min_losses)
    std_losses = np.std(min_losses)
    conf_interval_max = avg_losses + 1.96*(std_losses/np.sqrt(range_val))
    conf_interval_min = avg_losses - 1.96*(std_losses/np.sqrt(range_val))
    print(f"K:{K}, Average Loss:{avg_losses}, Confidence Interval:[{conf_interval_min},{conf_interval_max}]")
    averages_2[index] = avg_losses
    index += 1

plt.plot(k_values, averages_2)
plt.title('Dataset 2 with Kmeans++') # Title of the plot
plt.xlabel('K values') # X-Label
plt.ylabel('Loss value') # Y-Label
plt.show()