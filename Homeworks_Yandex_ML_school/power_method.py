def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE

    eigenvector = np.random.rand(len(data))
    for _ in range(num_steps):
        eigenvector = np.dot(data, eigenvector)
        eigenvector /= np.linalg.norm(eigenvector)

    eigenvalue = float(np.dot(np.dot(eigenvector, data), eigenvector))
    return eigenvalue, eigenvector
