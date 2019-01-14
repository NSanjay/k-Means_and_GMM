__author__ = 'Sanjay Narayana'

import numpy as np
from numpy import linalg as la
from k_means import KMeans
#from Test import k_means

np.random.seed(10)

class PCA(object):
    def __init__(self):
        print ("brooo")
        self.number_of_principal_components = 2

    def compute_principal_components(self):
        self.audio_data = np.genfromtxt('audioData.csv', delimiter=',')
        mean_of_audio_data = np.mean(self.audio_data,axis=0)
        centered_data = self.audio_data - mean_of_audio_data
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigen_values, eigen_vectors = la.eigh(covariance_matrix)
        print("vals:::",eigen_values)
        print("vectors::",eigen_vectors,eigen_vectors.shape)
        eigen_values_sorted=sorted(eigen_values,reverse=True)
        print("vals:::", eigen_values_sorted)
        indices_of_top_eigen_vectors = np.where(eigen_values >= eigen_values_sorted[self.number_of_principal_components-1])
        #print("lol::",lol)
        top_eigen_vectors = eigen_vectors[indices_of_top_eigen_vectors]
        print("top_eigen_vectors",top_eigen_vectors)
        return top_eigen_vectors

    def project_data_along_principal_components(self,eigen_vectors):
        projected_data = np.dot(self.audio_data,eigen_vectors.T)
        return projected_data

if __name__ == '__main__':
    pca = PCA()
    eigen_vectors = pca.compute_principal_components()
    projected_data = pca.project_data_along_principal_components(eigen_vectors)
    clusters = range(2, 11)
    k_means = KMeans()

    #k_means(2,projected_data)

    losses = k_means.cluster(clusters,projected_data)
    k_means.plot_objective_function(clusters, losses)