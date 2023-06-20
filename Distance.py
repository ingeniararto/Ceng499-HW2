import numpy as np
from numpy import dot, sum
from numpy.linalg import norm


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        return 1-(dot(x, y)/(norm(x)*norm(y))) # in order to get dissimilarityn
    
    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        return sum(np.abs(x - y)**p)**(1/p)

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        difference = np.subtract(x, y)
        return np.sqrt(np.matmul(np.matmul(np.transpose(difference), S_minus_1), difference))

