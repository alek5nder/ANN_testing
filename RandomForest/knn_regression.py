import numpy as np

class KNNRegressor:
    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Custom K-Nearest Neighbors Regressor

        Parameters:
        - n_neighbors: number of neighbors to use
        - weights: 'uniform' or 'distance'
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """Predict target values for input samples"""
        X = np.array(X)
        predictions = []

        for x in X:
            # Compute distances to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of k nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_values = self.y_train[neighbor_indices]
            neighbor_distances = distances[neighbor_indices]

            if self.weights == 'uniform':
                # Simple average
                prediction = np.mean(neighbor_values)
            elif self.weights == 'distance':
                # Weighted average (inverse distance)
                # Avoid division by zero
                weights = 1 / (neighbor_distances + 1e-8)
                prediction = np.sum(weights * neighbor_values) / np.sum(weights)
            else:
                raise ValueError("weights must be 'uniform' or 'distance'")

            predictions.append(prediction)

        return np.array(predictions)
