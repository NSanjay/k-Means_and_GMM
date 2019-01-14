__author__ = 'Sanjay Narayana'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

np.random.seed(10)
random.seed(10)

class GaussianMixtureModels(object):
    def __init__(self):
        self.number_of_clusters = 2
        self.prior_probabilities = self.number_of_clusters * [1 / 2]
        self.threshold = 1e-6

    def distance_from_centers(self, centroids, data):
        D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in data])
        # print("D2:::",D2)
        return D2

    def choose_next_center(self, D2, data):
        probabilities = D2 / D2.sum()
        # print("probabilities::",probabilities)
        cumulative_probabilities = probabilities.cumsum()
        # print("cum::",cumulative_probabilities)
        r = random.random()
        print("r::", r)
        ind = np.where(cumulative_probabilities >= r)[0][0]
        print("ind::", ind, np.array([ind]))
        print("sample::", data[12].shape)
        return data[np.array([ind])]

    def init_centers(self, k, data):
        # centroids = random.sample(self.audio_data, 1)
        # centroids = np.random.choice(self.audio_data, 1,replace=False)
        random_centroid = np.random.choice(data.shape[0], 1, replace=False)
        # random_centroid = np.array([4])
        # random_centroid = np.array([11]) # best until now
        # random_centroid = np.array([17])
        print("np rand:::", random_centroid)
        centroids = data[random_centroid]

        # print("centroids::",centroids.shape)
        while len(centroids) < k:
            D2 = self.distance_from_centers(centroids, data)
            an_array = self.choose_next_center(D2, data)
            print("shape:::", an_array.shape)
            centroids = np.append(centroids, an_array, axis=0)
            print("centroids::", centroids.shape)
        return centroids

    def read_data(self):
        self.audio_data = np.genfromtxt('audioData.csv', delimiter=',')
        #np.random.shuffle(self.audio_data)
        self.data_shape = self.audio_data.shape
        self.memberships = self.audio_data.shape[0] * [None]

    def initialize_parameters(self):
        # indices = np.random.choice(self.audio_data.shape[0], self.number_of_clusters, replace=False)
        #indices = np.array([60, 120])
        #indices = np.array([69, 120])
        #indices = np.array([37, 120])
        #print("indices:::", indices)

        #self.centers = self.audio_data[indices]
        self.centers = self.init_centers(2, self.audio_data)
        self.covariance_matrix = np.cov(self.audio_data, rowvar=False)

    def calculate_likelihoods(self):
        likelihoods = np.empty((2, 128))
        for i in range(self.number_of_clusters):
            pdf = multivariate_normal.pdf(self.audio_data, mean=self.centers[i], cov=self.covariance_matrix)
            # log_pdf = multivariate_normal.logpdf(self.audio_data, mean=self.centers[i],cov=self.covariance_matrix)
            likelihoods[i] = pdf
            # log_likelihoods[i] = log_pdf
        #print("likelihoods:::", likelihoods.shape)
        return likelihoods.T

    def em_algorithm(self):
        previous_log_likelihood = 0
        #while True: 151
        for a in range(9):
            likelihoods = self.calculate_likelihoods()
            print("likelihoods:::", likelihoods)
            likelihoods_and_prior_probabilities = likelihoods * self.prior_probabilities
            point_probabilities = np.sum(likelihoods_and_prior_probabilities, axis=1)
            print("point_probabilities_shape:::", likelihoods_and_prior_probabilities.shape,
                  point_probabilities[:, None].shape)
            log_of_point_probabilities = np.log(point_probabilities)
            log_likelihood_of_data = np.sum(log_of_point_probabilities)
            normalized_scores = np.divide(likelihoods_and_prior_probabilities,
                                          point_probabilities[:, None])

            comparisons = normalized_scores[:, 0] >= normalized_scores[:, 1]
            print("normalized_shapes:::", normalized_scores.shape)
            print("log_likelihood:::", log_likelihood_of_data)
            if np.abs(log_likelihood_of_data - previous_log_likelihood) < self.threshold:
                #break
                pass
            previous_log_likelihood = log_likelihood_of_data
            for i in range(len(self.centers)):
                sum_of_dimension_values = np.sum(normalized_scores[:, i])
                print("extended:::", normalized_scores[:, i][:, None].shape)
                self.centers[i] = np.sum((normalized_scores[:, i][:, None] * self.audio_data),
                                         axis=0) / sum_of_dimension_values
                self.prior_probabilities[i] = sum_of_dimension_values / self.data_shape[0]
        print("i::",a)
        return comparisons

    def plot_scatter(self, comparisons):
        print("comparisons::", comparisons)
        # print("not comparisopns::",np.logical_not(comparisons))
        _1st_cluster_points = self.audio_data[comparisons]
        _2nd_cluster_points = self.audio_data[np.logical_not(comparisons)]
        figure = plt.figure(1)
        axes = figure.add_subplot(111)
        plt.title("GMM Clustering of Audio Data")
        plt.xlabel("1st Feature")
        plt.ylabel("2nd Feature")
        axes.scatter(_1st_cluster_points[:, 0], _1st_cluster_points[:, 1], c='b', marker='.', label='First Cluster')
        axes.scatter(_2nd_cluster_points[:, 0], _2nd_cluster_points[:, 1], c='r', marker='.', label='Second Cluster')
        plt.legend(loc='upper left')
        plt.show()


if __name__ == '__main__':
    gmm = GaussianMixtureModels()
    gmm.read_data()
    gmm.initialize_parameters()
    # gmm.compute_likelihood()
    comparisons = gmm.em_algorithm()
    # gmm.calculate_likelihood()
    gmm.plot_scatter(comparisons)
