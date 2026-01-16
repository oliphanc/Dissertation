import numpy as np
from pathlib import Path
from numba import njit
import joblib

from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import KDTree

from PytorchModels import AutoEncoder, NNRegressor

@njit
def MinObjective(w, y_neighbors, ystar, eps, gamma):
    
    N, d = y_neighbors.shape
    y_est = np.dot(w, y_neighbors)

    diffs = y_neighbors - ystar
    G = np.dot(diffs, diffs.T)
    dists_squared = np.sum(diffs ** 2, axis=1)
    max_dist_squared = np.max(dists_squared)
    
    if max_dist_squared == 0:
        normalized_dists = np.zeros_like(dists_squared)
    else:
        normalized_dists = dists_squared / max_dist_squared

    c = eps * normalized_dists ** gamma
    return (np.linalg.norm(y_est - ystar) + np.linalg.norm(c))


class ReducedOrderModel:
    def __init__(self, n_components, method='PCA', 
                 regressor = GaussianProcessRegressor(), 
                 k_neighbors=5, epsilon=0.01, k_penalty=4,
                 ae_kwargs=None, nn_kwargs=None):
        """
        Initialize the reduced-order model with a chosen dimensionality reduction method.
        :param n_components: Number of components for dimensionality reduction
        :param method: Dimensionality reduction method ('PCA', 'ISOMAP', 'AE')
        :param regressor: Regression object to be used in predictions ('NN' or scikit-learn models)
        :param k_neighbors: Number of nearest neighbors for backmapping
        :param epsilon: Small regularization factor for weight computation
        :param k_penalty: Exponent for distance-based penalty term in weight calculation
        :param ae_kwargs: Keyword arguments for AutoEncoder if method is 'AE'
        :param nn_kwargs: Keyword arguments for NNRegressor if method is 'NN'
        """
        self.n_components = n_components
        self.method = method.upper()
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon
        self.k_penalty = k_penalty
        self.scaler = StandardScaler()

        # ---------------------- Dimensionality reducer -----------------------
        if self.method == 'PCA':
            self.reducer = PCA(n_components=n_components)
            self.backmapper = None  # PCA has exact inverse
        elif self.method == 'ISOMAP':
            self.reducer = Isomap(n_neighbors=k_neighbors, n_components=n_components)
            self.backmapper_data = None
        elif self.method == 'LLE':
            self.reducer = LocallyLinearEmbedding(n_components=n_components)
            self.backmapper_data = None
        elif self.method == 'AE':
            ae_kwargs = ae_kwargs or {}
            self.reducer = AutoEncoder(n_components, **ae_kwargs)
        else:
            raise ValueError("Unsupported reduction method. Choose 'PCA', 'ISOMAP', or 'AE'.")

        # --------------------------- Regressor -------------------------------
        if regressor is None or regressor == 'NN':
            nn_kwargs = nn_kwargs or {}
            self.regressor = NNRegressor(**nn_kwargs)
        else:
            self.regressor = regressor
    
    def fit(self, X, Y):
        """
        Fit the reduced-order model.
        :param X: Input features
        :param Y: High-dimensional target values
        """
        Y_scaled = self.scaler.fit_transform(Y)
        Y_reduced = self.reducer.fit_transform(Y_scaled)
        self.regressor.fit(X, Y_reduced)
        
        # Store data for backmapping
        if self.method == 'ISOMAP':
            self.backmapper_data = (Y_reduced, Y_scaled)
            self.low_tree = KDTree(Y_reduced, leaf_size=self.k_neighbors)
    
    def compute_weights(self, y_star, Y_neighbors):
        """
        Compute reconstruction weights using distance-based penalty as described in the paper.
        """     
        
        w_init = np.ones(self.k_neighbors) / self.k_neighbors
        res = minimize(MinObjective, w_init, args=(Y_neighbors, y_star, self.epsilon, self.k_penalty),
                       constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                       method='SLSQP', options={'maxiter': 300})
        
        return res.x
    
    def backmap(self, Y_reduced_pred):
        """
        Apply backmapping using locally weighted reconstruction.
        """
        Y_reduced, Y_scaled = self.backmapper_data
        Y_pred_scaled = np.zeros((Y_reduced_pred.shape[0], Y_scaled.shape[1]))
        
        for i, y_star in enumerate(Y_reduced_pred):
            _, neighbor_indices = self.low_tree.query(y_star.reshape(1, -1), k=self.k_neighbors)
            neighbor_indices = neighbor_indices.squeeze(0)
            Y_neighbors = Y_reduced[neighbor_indices]
            W_neighbors = Y_scaled[neighbor_indices]
            weights = self.compute_weights(y_star, Y_neighbors)
            Y_pred_scaled[i] = np.sum(weights[:, None] * W_neighbors, axis=0)
        
        return Y_pred_scaled
    
    def predict(self, X):
        """
        Predict using the reduced-order model.
        :param X: Input features
        :return: Predicted high-dimensional target values
        """
        Y_reduced_pred = self.regressor.predict(X)
        
        if self.method in ['PCA', 'AE']:
            Y_pred_scaled = self.reducer.inverse_transform(Y_reduced_pred)
        else:
            Y_pred_scaled = self.backmap(Y_reduced_pred)
        
        return self.scaler.inverse_transform(Y_pred_scaled)
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimates using the regressor's predict method.
        :param X: Input features
        :param n_samples: Number of samples for uncertainty estimation
        :return: Mean and standard deviation of predictions
        """

        if self.method != "PCA":
            raise RuntimeError("Uncertainty is only implemented for PCA reduction.")
        if not isinstance(self.regressor, GaussianProcessRegressor):
            raise RuntimeError(
                "Uncertainty requires GaussianProcessRegressor as the regressor."
            )
        

        Z_mean, Z_std= self.regressor.predict(X, return_std=True)
        
        W = self.reducer.components_.T  # (n_features, n_components)
        mu_pca = self.reducer.mean_      # (n_features,)

        # mean in scaled space
        Y_scaled_mean = Z_mean @ W.T + mu_pca          # (n_samples, n_features)

        W_sq = self.reducer.components_.T**2            # Wᵀ element-wise²

        # scaled variance:  σ_y² = σ_z² · W²ᵀ
        Y_var_scaled = Z_std**2 @ W_sq.T                   # (n_samples, n_features)

        # undo StandardScaler (variance scales by scale², std by scale)
        scale = self.scaler.scale_                      # (n_features,)
        Y_std = np.sqrt(Y_var_scaled) * scale
        Y_mean = Y_scaled_mean * scale + self.scaler.mean_
        
        return Y_mean, Y_std
    
    def save(self, path: str | Path, compress: int | bool = 3) -> None:
        """
        Persist the entire trained model to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination filename (e.g. ``'my_rom.pkl'`` or ``'my_rom.joblib'``).
        compress : int | bool, default 3
            Joblib compression level; 0/False disables compression,
            1-9 choose gzip levels (3 is a good speed/size trade-off).
        """
        joblib.dump(self, path, compress=compress)


    @classmethod
    def load(cls, path: str | Path):
        """
        Restore a saved ReducedOrderModel from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            File created with :py:meth:`save`.

        Returns
        -------
        ReducedOrderModel
        """
        return joblib.load(path)
    

if __name__ == "__main__":
    # Example usage
    
    from sklearn.datasets import make_regression
    xtrain, ytrain = make_regression(n_samples=1000, n_features=10, n_targets=100, noise=0.1, random_state=42)

    regressor = GaussianProcessRegressor(alpha=1e-8, n_restarts_optimizer=0)
    rom = ReducedOrderModel(n_components=.90, method='PCA', regressor=regressor, k_neighbors=150)

    yhat = ytrain.reshape(ytrain.shape[0], -1)
    rom.fit(xtrain, yhat)