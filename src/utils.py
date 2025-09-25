"""Utility functions for device detection and configuration."""

import torch
import numpy as np
from typing import Optional, Union


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        force_cpu: If True, force CPU usage even if CUDA is available

    Returns:
        torch.device: Either 'cuda' or 'cpu'
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_vectors(
    vectors: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize vectors to unit length.

    Args:
        vectors: Input vectors of shape (n_vectors, dimension)

    Returns:
        Normalized vectors of the same shape and type
    """
    if isinstance(vectors, np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        return vectors / norms
    else:
        return torch.nn.functional.normalize(vectors, p=2, dim=1)


def compute_max_abs_cosine(vectors: np.ndarray) -> float:
    """
    Compute the maximum absolute cosine similarity between any pair of vectors.

    Args:
        vectors: Normalized vectors of shape (n_vectors, dimension)

    Returns:
        Maximum absolute cosine similarity (epsilon)
    """
    if len(vectors) <= 1:
        return 0.0

    # Normalize vectors
    vectors = normalize_vectors(vectors)

    # Compute cosine similarity matrix
    cos_sim_matrix = np.dot(vectors, vectors.T)

    # Get upper triangular part (excluding diagonal)
    n = len(vectors)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    similarities = np.abs(cos_sim_matrix[mask])

    return float(np.max(similarities)) if len(similarities) > 0 else 0.0
