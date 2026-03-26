import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    # Convert to numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Handle 1D inputs
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Compute squared Euclidean distances (no sqrt needed)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)  # shape: (n_test, n_train)

    # Sort distances and get indices
    sorted_indices = np.argsort(dist_sq, axis=1)

    # Prepare output with padding if k > n_train
    result = np.full((n_test, k), -1, dtype=int)

    # Number of valid neighbors we can fill
    valid_k = min(k, n_train)

    # Fill valid indices
    result[:, :valid_k] = sorted_indices[:, :valid_k]

    return result