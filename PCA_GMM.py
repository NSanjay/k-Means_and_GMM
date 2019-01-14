__author__ = 'Sanjay Narayana'

import numpy as np
from numpy import linalg as la
import random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
#from Test import k_means

np.random.seed(10)
random.seed(10)

class PCA_GMM(object):
    def __init__(self):
        print ("brooo")
        self.number_of_principal_components = 2
        self.number_of_clusters = 2
        self.prior_probabilities = self.number_of_clusters * [1 / 2]
        self.threshold = 1e-6

    def compute_principal_components(self):
        self.audio_data = np.genfromtxt('audioData.csv', delimiter=',')
        mean_of_audio_data = np.mean(self.audio_data,axis=0)
        centered_data = self.audio_data - mean_of_audio_data
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigen_values, eigen_vectors = la.eig(covariance_matrix)
        print("vals:::",eigen_values)
        #print("vectors::",eigen_vectors,eigen_vectors.shape)
        eigen_values_sorted=sorted(eigen_values,reverse=True)
        #print("vals:::", eigen_values_sorted)
        indices_of_top_eigen_vectors = np.where(eigen_values >= eigen_values_sorted[self.number_of_principal_components-1])
        #print("lol::",lol)
        top_eigen_vectors = eigen_vectors[indices_of_top_eigen_vectors]
        print("top_eigen_vectors",top_eigen_vectors)
        return top_eigen_vectors

    def project_data_along_principal_components(self,eigen_vectors):
        projected_data = np.dot(self.audio_data,eigen_vectors.T)
        self.audio_data = projected_data
        self.data_shape = self.audio_data.shape
        return projected_data

    def distance_from_centers(self,centroids,data):
        D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in data])
        #print("D2:::",D2)
        return D2

    def choose_next_center(self,D2,data):
        probabilities = D2 / D2.sum()
        #print("probabilities::",probabilities)
        cumulative_probabilities = probabilities.cumsum()
        #print("cum::",cumulative_probabilities)
        r = random.random()
        #r = np.random.rand(1)[0]
        print("r::",r)
        ind = np.where(cumulative_probabilities >= r)[0][0]
        print("ind::",ind,self.audio_data[ind])
        print("sample::",data[12].shape)
        return data[np.array([ind])]

    def init_centers(self,k,data):
        #centroids = random.sample(self.audio_data, 1)
        #centroids = np.random.choice(self.audio_data, 1,replace=False)

        np.random.seed(15)
        random.seed(15)
        random_centroid = np.random.choice(data.shape[0], 1, replace=False)
        #random_centroid = np.array([4])
        #random_centroid = np.array([11]) # best until now
        #random_centroid = np.array([17])
        print("np rand:::",random_centroid)
        centroids = data[random_centroid]

        print("centroids::",centroids)
        while len(centroids) < k:
            D2 = self.distance_from_centers(centroids,data)
            an_array = self.choose_next_center(D2,data)
            print("shape:::",an_array.shape)
            centroids = np.append(centroids,an_array,axis=0)
            print("centroids::", centroids.shape)
        return centroids

    def initialize_parameters(self):
        #indices = np.random.choice(self.audio_data.shape[0], self.number_of_clusters, replace=False)
        #indices = np.array([60, 120])
        #indices = np.array([4, 2])
        #indices = np.array([37, 12])

        #indices = np.array([69, 120])
        #self.centers = self.audio_data[indices]

        self.centers = self.init_centers(2, self.audio_data)
        #self.centers = np.array([[68.082,-12.7102],[85.8203,-17.1969]])
        #self.centers = np.array([[85.8203, -17.1969], [85.8203, -17.1969]])
        print("centr::", self.centers)
        self.covariance_matrix = np.cov(self.audio_data, rowvar=False)
        print("dim::",self.covariance_matrix.shape)

    def calculate_likelihoods(self):
        likelihoods = np.empty((2,128))
        for i in range(self.number_of_clusters):
            pdf = multivariate_normal.pdf(self.audio_data, mean=self.centers[i],cov=self.covariance_matrix)
            #log_pdf = multivariate_normal.logpdf(self.audio_data, mean=self.centers[i],cov=self.covariance_matrix)
            #print("likelihood::",i,likelihoods[i])
            likelihoods[i] = pdf
            #print("likelihood_after::", i, likelihoods[i])
            #log_likelihoods[i] = log_pdf
        #print("likelihoods:::", likelihoods.shape)
        return likelihoods.T

    def em_algorithm(self):
        previous_log_likelihood = 0
        #while True:
        for _ in range(30):
            likelihoods = self.calculate_likelihoods()
            print("likelihoods:::",likelihoods)
            likelihoods_and_prior_probabilities = likelihoods * self.prior_probabilities
            point_probabilities = np.sum(likelihoods_and_prior_probabilities, axis=1)
            #print("point_probabilities_shape:::",likelihoods_and_prior_probabilities.shape,point_probabilities[:,None].shape)
            log_of_point_probabilities = np.log(point_probabilities)
            log_likelihood_of_data = np.sum(log_of_point_probabilities)
            normalized_scores = np.divide(likelihoods_and_prior_probabilities,
                                          point_probabilities[:, None])

            comparisons = normalized_scores[:, 0] >= normalized_scores[:, 1]
            #print("normalized_shapes:::",normalized_scores.shape)
            #print("log_likelihood:::",log_likelihood_of_data)
            if np.abs(log_likelihood_of_data - previous_log_likelihood) < self.threshold:
                break
            previous_log_likelihood=log_likelihood_of_data
            for i in range(len(self.centers)):
                sum_of_dimension_values = np.sum(normalized_scores[:, i])
                #print("extended:::",normalized_scores[:,i][:,None].shape)
                self.centers[i] = np.sum((normalized_scores[:,i][:,None] * self.audio_data), axis=0) / sum_of_dimension_values
                self.prior_probabilities[i] = sum_of_dimension_values / self.data_shape[0]
                #print("prior_prob::",self.prior_probabilities[i])
        return comparisons

    def plot_scatter(self,comparisons):
        print("comparisons::",comparisons)
        #print("not comparisopns::",np.logical_not(comparisons))
        print("sh::",self.audio_data.shape)
        _1st_cluster_points = self.audio_data[comparisons]
        _2nd_cluster_points = self.audio_data[np.logical_not(comparisons)]
        figure = plt.figure(1)
        axes = figure.add_subplot(111)
        plt.title("GMM Clustering of Audio Data")
        plt.xlabel("1st Feature")
        plt.ylabel("2nd Feature")
        axes.scatter(_1st_cluster_points[:,0],_1st_cluster_points[:,1],c='b',marker='.',label='First Cluster')
        axes.scatter(_2nd_cluster_points[:,0], _2nd_cluster_points[:,1],c='r',marker='.',label='Second Cluster')
        plt.legend(loc='lower left')
        plt.show()

if __name__ == '__main__':
    pca = PCA_GMM()
    eigen_vectors = pca.compute_principal_components()
    projected_data = pca.project_data_along_principal_components(eigen_vectors)
    pca.initialize_parameters()
    # gmm.compute_likelihood()
    comparisons = pca.em_algorithm()
    # gmm.calculate_likelihood()
    pca.plot_scatter(comparisons)
