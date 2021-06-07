########################################################################
#                                                                      #
#                    Nearest Local Centroid Classifier                 #
#                        Author: Jeremy Walker                         #                        
#   Author Affiliation: San Jose State University College of Science   #
#                     Date Last Modified: 6/21/2021                    #     
#                     E-mail: jeremy.walker@sjsu.edu                   #       
#                                                                      #     
########################################################################

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

#X_train should be the numeric training data in a numpy matrix, with each row corresponding 
#to a data point
#y_train should be a 1-dimensional numpy array of integers to code for each class
#query_vec should be a 1-dimensional (row) numpy array to be classified based on training data 
#k is the number of nearest neighbors to be considered when calculating the local class centroids
def NLC(X_train, y_train, query_vec, k):
    centroidDists = []; n_classes = len(set(y_train))
    for j in range(n_classes):
        classSubset = X_train[y_train == j,:]
        neighbors = NearestNeighbors(n_neighbors = k)
        neighbors.fit(classSubset)
        distances, indices = neighbors.kneighbors(np.reshape(query_vec,(1,-1)))
        centroid = np.mean(classSubset[indices,:], axis = 1)
        dist = distance.euclidean(centroid, query_vec)
        centroidDists.append(dist)
    return np.argmin(centroidDists)
#this function will return the classification of a single query vector; included below is an 
#example implementation to apply NLC to a matrix of multiple data points for classification

y_pred = [NLC(X_train, y_train, query_vec = i, k = 9) for i in X_test]


