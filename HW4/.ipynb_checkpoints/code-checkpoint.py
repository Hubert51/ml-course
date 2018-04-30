import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def genObservations(pi, num_of_samples, means, covariances, seed=0):
    num_of_samples_class = [int(i*num_of_samples) for i in pi]
    num_of_classes = len(means)
    np.random.seed(seed)
    generated_samples = []
    for i in range(num_of_classes):
        samples = np.random.multivariate_normal(means[i], covariances[i], num_of_samples_class[i])
        generated_samples.extend(list(samples))
    return np.array(generated_samples)

def plotScatter(data):
    plt.figure(figsize=(10,10))
    plt.scatter(data[:,0], data[:,1])
    plt.xlabel("X-axis", fontsize=20)
    plt.ylabel("Y-axis", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(False)


class KMeans():

    def __init__(self, K, seed=0):
        self.K = K
        self.C = None
        self.seed = seed
        np.random.seed(seed)

    def setData(self, data):
        self.data = data
        (self.n, self.d) = data.shape
        self.C = np.empty(self.n)
        self.centroids = data[np.random.choice(self.n, size=self.K, replace=False), :]

    def calcLoss(self):
        Loss = 0
        for n in range(self.n):
            Loss += (np.linalg.norm(self.data[n, :] - self.centroids[int(self.C[n]), :]) ** 2)
        return Loss

    def updateClusters(self):
        for p in range(self.n):  # for every point
            D = np.Inf
            for k in range(self.K):  # check every cluster
                d = (np.linalg.norm(self.data[p, :] - self.centroids[k, :]) ** 2)
                if d < D:
                    self.C[p] = k
                    D = d

    def updateCentroids(self):
        for k in range(self.K):
            self.centroids[k] = np.mean(self.data[self.C == k, :], axis=0)

    def cluster(self, converge_condition=None):
        K = self.K
        data = self.data
        self.Loss = []
        if converge_condition != None:
            for i in range(converge_condition):
                self.updateClusters()
                self.updateCentroids()
                self.Loss.append(self.calcLoss())

        else:
            loss = 0
            while ((loss - self.calcLoss()) > 1):
                loss = self.calcLoss()
                self.updateClusters()
                self.updateCentroids()
                self.Loss.append(loss)

        self.setClusters()

    def setClusters(self):
        self.clusters = []
        for k in range(self.K):
            self.clusters.append(self.data[self.C == k, :])

    def plotLoss(self):
        sns.set_style('whitegrid')
        sns.set_context('poster')
        plt.figure(figsize=(30, 10))
        plt.plot(range(1, 21), self.Loss)
        plt.xlabel("Iterations", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.title('Loss plot for K = {}'.format(self.K), fontsize=20)
        plt.xticks(range(1, 21), fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    def plotScatter(self):
        sns.set_style('whitegrid')
        plt.figure(figsize=(10, 10))
        for k in range(self.K):
            plt.scatter(self.clusters[k][:, 0], self.clusters[k][:, 1], label="Cluster# {}".format(k + 1))
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color="black", marker="*", s=288)
        plt.xlabel("X-axis", fontsize=20)
        plt.ylabel("Y-axis", fontsize=20)
        plt.title('Scatter plot for K = {}'.format(self.K), fontsize=20)
        plt.grid(False)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.show()


def testPart1a(data):
    for k in [2, 3, 4, 5]:
        KM = KMeans(k)
        KM.setData(data)
        KM.cluster(20)
        KM.plotLoss()


def testPart1b(data):
    for k in [3, 5]:
        KM = KMeans(k)
        KM.setData(data)
        KM.cluster(20)
        KM.plotScatter()

if __name__ == '__main__':
    m1   = np.array([0,0])
    cov1 = np.array([[1,0],
                     [0,1]])

    m2   = np.array([3,0])
    cov2 = np.array([[1,0],
                     [0,1]])

    m3   = np.array([0,3])
    cov3 = np.array([[1,0],
                     [0,1]])

    data = genObservations([0.2, 0.5, 0.3], 500, [m1, m2, m3], [cov1, cov2, cov3])

    testPart1a(data)

    testPart1b(data)

