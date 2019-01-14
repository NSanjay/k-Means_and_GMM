__author__ = 'Sanjay Narayana'

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(10)

class KMeans(object):
    def __init__(self):
        '''The algorithm has converged when the assignments no longer change.
        There is no guarantee that the optimum is found using this algorithm.
        '''
        print ("brooo")

    def distance_from_centers(self,centroids,data):
        D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in data])
        print("D2:::",D2)
        return D2

    def choose_next_center(self,D2,data):
        probabilities = D2 / D2.sum()
        print("probabilities::",probabilities)
        cumulative_probabilities = probabilities.cumsum()
        print("cum::",cumulative_probabilities)
        r = random.random()
        print("r::",r)
        ind = np.where(cumulative_probabilities >= r)[0][0]
        print("ind::",ind,np.array([ind]))
        print("sample::",data[12].shape)
        return data[np.array([ind])]

    def init_centers(self,k,data):
        #centroids = random.sample(self.audio_data, 1)
        #centroids = np.random.choice(self.audio_data, 1,replace=False)
        random_centroid = np.random.choice(data.shape[0], 1, replace=False)
        #random_centroid = np.array([11])
        print("np rand:::",random_centroid)
        centroids = data[random_centroid]

        #print("centroids::",centroids.shape)
        while len(centroids) < k:
            D2 = self.distance_from_centers(centroids,data)
            an_array = self.choose_next_center(D2,data)
            print("shape:::",an_array.shape)
            centroids = np.append(centroids,an_array,axis=0)
            print("centroids::", centroids.shape)
        return centroids


    def read_data(self):
        self.audio_data = np.genfromtxt('audioData.csv',delimiter=',')
        self.memberships = self.audio_data.shape[0] * [None]

    def k_means(self, centers, data):
        loss_value = 0
        previous_loss_value = 0
        previous_centers = np.random.randint(centers.shape[0],size=data.shape[0])
        condition = True
        while condition:

            #This statement might cause memory error for
            #higher number of cluster centers or for
            #memory constrained systems
            expanded_audio_data = data[:,np.newaxis,:]
            differences = expanded_audio_data - centers
            squared_differences = np.einsum('ijk,ijk->ij', differences, differences)
            print("")
            #squared_differences = np.sqrt(squared_differences)
            min_centers = np.argmin(squared_differences,axis=1)

            # if np.array_equal(min_centers, previous_centers):
            #     condition = False

            #print("distances::",np.amin(squared_differences,axis=1))
            loss_value = np.sum(np.amin(squared_differences,axis=1))
            if abs(loss_value - previous_loss_value) < 1e-6 and np.array_equal(min_centers, previous_centers):
                condition = False
            previous_loss_value = loss_value
            previous_centers = min_centers
            for i in range(len(centers)):
                centers[i] = np.nanmean(data[min_centers == i],axis=0)
        return loss_value


    def cluster(self, clusters, data):
        losses = []
        for cluster in clusters:
            #indices = np.array([1, 111])
            indices = np.random.choice(data.shape[0], cluster, replace=False)
            if cluster is 2:
                indices = np.array([69, 120])
            #centers = self.init_centers(cluster,data)
            centers = data[indices]
            loss_value = self.k_means(centers,data)
            losses.append(loss_value)
        return losses

    def plot_objective_function(self, clusters, losses):
        plt.plot(clusters, losses)
        plt.title('Number of Clusters v/s Loss Function')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Loss Function')
        plt.show()

if __name__ == '__main__':
    k_means = KMeans()
    k_means.read_data()
    clusters = range(2,11)
    losses = k_means.cluster(clusters,k_means.audio_data)
    k_means.plot_objective_function(clusters,losses)