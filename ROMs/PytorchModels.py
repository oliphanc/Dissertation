import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class _MLP(nn.Module):
    """Simple fully‑connected feed‑forward network (double precision)."""
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden) + [out_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        # Ensure parameters are double precision regardless of global default
        self.double()

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 1. Autoencoder reducer (acts like scikit‑learn transformer)
# -----------------------------------------------------------------------------

class AutoEncoder:
    """Non‑linear dimensionality reducer based on an autoencoder (AE)."""
    def __init__(self, n_components, hidden=(512, 128), epochs=400, batch_size=256,
                 lr=1e-3, device=None, verbose=False):
        self.n_components = n_components
        self.hidden = list(hidden)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.decoder = None

    # ------------------------------------------------------------------
    def _build_networks(self, input_dim):
        """Construct symmetric encoder/decoder in double precision."""
        self.encoder = _MLP(input_dim, self.n_components, self.hidden).to(self.device)
        self.decoder = _MLP(self.n_components, input_dim, list(reversed(self.hidden))).to(self.device)

    # ------------------------------------------------------------------
    def fit_transform(self, Y):
        Y = np.asarray(Y, dtype=np.float64)
        n_samples, n_features = Y.shape
        self._build_networks(n_features)

        dataset = TensorDataset(torch.from_numpy(Y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        mse = nn.MSELoss()
        losses = []
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                encoded = self.encoder(batch)
                decoded = self.decoder(encoded)
                loss = mse(decoded, batch)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            losses.append(total_loss / len(loader))
            if self.verbose and (epoch % (self.epochs//10) == 0):
                print(f"AE‑epoch {epoch:4d} — loss: {total_loss / n_samples:.6e}")

        # Cache decoder for inverse transform
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            Y_reduced = self.encoder(torch.from_numpy(Y).to(self.device)).cpu().numpy()
        
        if self.verbose:
            print(f"Final AE loss: {losses[-1]:.6e}")
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Autoencoder Training Loss")
            plt.show()

        return Y_reduced

    # ------------------------------------------------------------------
    def transform(self, Y):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(torch.from_numpy(Y.astype(np.float64)).to(self.device)).cpu().numpy()

    def inverse_transform(self, Z):
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder(torch.from_numpy(Z.astype(np.float64)).to(self.device)).cpu().numpy()

# -----------------------------------------------------------------------------
# 2. Neural‑network regressor with scikit‑learn like API
# -----------------------------------------------------------------------------

class NNRegressor:
    """Feed‑forward neural‑network regressor (double precision)."""
    def __init__(self, hidden=(128, 128), epochs=800, batch_size=256, lr=1e-3,
                 device=None, verbose=False):
        self.hidden = list(hidden)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model = None

    def _build_model(self, in_dim, out_dim):
        self.model = _MLP(in_dim, out_dim, self.hidden).to(self.device)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n_samples, in_dim = X.shape
        out_dim = Y.shape[1]
        if self.model is None:
            self._build_model(in_dim, out_dim)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        losses = []
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = mse(pred, yb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            losses.append(total / len(loader))
            if self.verbose and (epoch % (self.epochs//10) == 0):
                print(f"NN‑epoch {epoch:4d} — loss: {total / n_samples:.6e}")
        self.model.eval()
        if self.verbose:
            print(f"Final NN loss: {losses[-1]:.6e}")
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Neural Network Training Loss")
            plt.show()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        with torch.no_grad():
            preds = self.model(torch.from_numpy(X).to(self.device)).cpu().numpy()
        return preds
