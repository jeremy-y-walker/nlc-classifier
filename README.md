# nlc-classifier
By-hand implementation of the nearest local centroid (NLC) classifier. Python's SciKit Learn library has a built-in implementation of the nearest centroid 
classifier, but the code in this repository incorporates a k-nearest neighbors approach which compares the Euclidean distance from a query data vector to the 
centroids of each class computed from the k-neighbors in each class. Classification is labeled as the class whose centroid is nearest said query vector.
