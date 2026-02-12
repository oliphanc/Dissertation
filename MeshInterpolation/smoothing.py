import numpy as np


def elliptical_smoothing(X, Y, max_iter=5000, tol=1e-4, alpha=0.5):
    """
    Laplacian smoothing of a structured grid using SOR.
    Boundary nodes remain fixed.
    """
    ny, nx = X.shape

    for _ in range(max_iter):
        X_old = X.copy()
        Y_old = Y.copy()

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                X[i, j] = 0.25 * (
                    X[i + 1, j] + X[i - 1, j] + X[i, j + 1] + X[i, j - 1]
                )
                Y[i, j] = 0.25 * (
                    Y[i + 1, j] + Y[i - 1, j] + Y[i, j + 1] + Y[i, j - 1]
                )

        diff = np.max(np.abs(X - X_old)) + np.max(np.abs(Y - Y_old))
        if diff < tol:
            break

    return X, Y
