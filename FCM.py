
import numpy as np
import pandas as pd
import copy
from caculationDistance import euclideanDistance, squaredNornal, relativeDist, distTwoSample
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import joblib


class FCM:
    def __init__(self, matrix_data, c_cluster=4, tol=1e-8, maxItem=1000):

        self.matrix_data = matrix_data
        self.c_cluster = c_cluster
        self.tol = tol
        self.maxItem = maxItem

    def initCenter(self, *args):

        n_samples, n_feature = self.matrix_data.shape
        center = np.zeros((n_cluster, n_feature))
        center[0] = self.matrix_data[np.random.randint(n_samples)]
        for i in range(1, n_cluster):
            distance_to_centers = euclideanDistance(self.matrix_data, center[[j for j in range(i)]], square=True)
            closed_distance = np.min(distance_to_centers, axis=1)
            denominator = closed_distance.sum()
            point = np.random.rand() * denominator
            be_choosed = np.searchsorted(np.cumsum(closed_distance), point)
            be_choosed = min(be_choosed, n_samples - 1)
            center[i] = self.matrix_data[be_choosed]
        return center

    def fit(self):
        n_samples, n_feature = self.matrix_data.shape
        self.center = self.initCenter()
        u = np.zeros((n_samples, n_cluster))
        for i in range(self.maxItem):
            dist_matrix_squared = euclideanDistance(self.matrix_data, self.center, square=True)
            np.maximum(dist_matrix_squared, 0.001, out=dist_matrix_squared)
            np.reciprocal(dist_matrix_squared, dtype=float, out=dist_matrix_squared)
            sum_row = dist_matrix_squared.sum(axis=1)
            np.reciprocal(sum_row, dtype=float, out=sum_row)
            oldU = copy.deepcopy(u)
            np.einsum("ij,i->ij", dist_matrix_squared, sum_row.T, out=u)
            u_shift = squaredNornal(u - oldU)
            print("Error:", u_shift)
            if u_shift < eps:
                print("Early stop")
                break
            u2 = u ** 2
            for i in range(u2.shape[1]):
                centroid = np.einsum("ij,i->ij", matrix_data, u2[:, i].T).sum(axis=0) / (u2[:, i].sum() + 0.001)
                self.center[i] = centroid
        self.labels = np.argmax(u, axis=1)
        return self.center, self.labels

    def getSse(self):
        sse_sum = 0
        for r, c in enumerate(self.labels):
            sse_sum += distTwoSample(self.center[c], matrix_data[r], square=True)

    def getSilhouette(self):
        silhouette_average = silhouette_score(self.matrix_data, self.labels)
        return silhouette_average



data = pd.read_csv('xxxxx/original_data.csv', encoding='utf-8')
data = data['Wind speed', 'Output power', 'Generator speed']
values = data.values.astype('float32')
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
scaled = pd.DataFrame(scaled)
matrix_data = np.asarray(scaled).astype(float)
n_cluster = 4
eps = 1e-5
myFcm = FCM(matrix_data, c_cluster=4, tol=eps, maxItem=100)
center, labels = myFcm.fit()
