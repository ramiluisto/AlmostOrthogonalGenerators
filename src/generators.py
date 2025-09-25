"""Vector generation methods for almost orthogonal vectors."""

import numpy as np
import torch
from typing import Optional, Dict, Any
from tqdm import tqdm
from .utils import get_device, normalize_vectors, set_seed


class BaseGenerator:
    """Base class for all vector generators."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    def generate(self, dimension: int, count: int, **kwargs) -> np.ndarray:
        """
        Generate vectors.

        Args:
            dimension: Vector dimension
            count: Number of vectors to generate
            **kwargs: Additional method-specific parameters

        Returns:
            Array of shape (count, dimension) with unit-normalized vectors
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed})"


class RandomSamplingGenerator(BaseGenerator):
    """
    Generate random unit vectors with optional oversampling and pruning.

    This method generates random vectors from a normal distribution,
    normalizes them, and optionally oversamples and prunes to improve
    the maximum absolute cosine similarity.
    """

    def __init__(self, seed: Optional[int] = None, oversampling_factor: float = 2.0):
        """
        Initialize random sampling generator.

        Args:
            seed: Random seed for reproducibility
            oversampling_factor: Factor to oversample before pruning (1.0 = no oversampling)
        """
        super().__init__(seed)
        self.oversampling_factor = max(1.0, oversampling_factor)

    def generate(self, dimension: int, count: int, verbose: bool = False) -> np.ndarray:
        """Generate random unit vectors with optional oversampling and pruning."""
        if self.seed is not None:
            set_seed(self.seed)

        # Generate initial vectors
        n_initial = int(count * self.oversampling_factor)
        vectors = np.random.randn(n_initial, dimension)
        vectors = normalize_vectors(vectors)

        # If no oversampling, return directly
        if self.oversampling_factor == 1.0 or n_initial == count:
            return vectors[:count]

        # Prune to target count
        vectors = self._prune_vectors(vectors, count, verbose)
        return vectors

    def _prune_vectors(
        self, vectors: np.ndarray, target_count: int, verbose: bool
    ) -> np.ndarray:
        """
        Prune vectors to target count by iteratively removing worst offenders.

        Args:
            vectors: Initial set of vectors
            target_count: Target number of vectors
            verbose: Show progress bar

        Returns:
            Pruned vectors
        """
        current_vectors = vectors.copy()
        n_remove = len(current_vectors) - target_count

        if verbose:
            pbar = tqdm(total=n_remove, desc="Pruning vectors")

        for _ in range(n_remove):
            # Compute cosine similarity matrix
            cos_sim = np.abs(np.dot(current_vectors, current_vectors.T))
            np.fill_diagonal(cos_sim, 0)

            # Find vector involved in highest similarity
            max_sim_idx = np.unravel_index(np.argmax(cos_sim), cos_sim.shape)

            # Count how many times each vector is involved in high similarities
            worst_counts = np.sum(cos_sim == cos_sim.max(), axis=0)
            worst_idx = np.argmax(worst_counts)

            # Remove worst vector
            current_vectors = np.delete(current_vectors, worst_idx, axis=0)

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        return current_vectors


class JohnsonLindenstraussGenerator(BaseGenerator):
    """
    Generate vectors using Johnson-Lindenstrauss random projection.

    This method projects random vectors from a higher-dimensional space
    to the target dimension, with optional oversampling and pruning.
    """

    def __init__(self, seed: Optional[int] = None, oversampling_factor: float = 2.0):
        """
        Initialize JL generator.

        Args:
            seed: Random seed for reproducibility
            oversampling_factor: Factor to oversample before projection
        """
        super().__init__(seed)
        self.oversampling_factor = max(1.0, oversampling_factor)

    def generate(self, dimension: int, count: int, verbose: bool = False) -> np.ndarray:
        """Generate vectors using JL projection."""
        if self.seed is not None:
            set_seed(self.seed)

        # Determine source dimension (higher than target)
        n_initial = int(count * self.oversampling_factor)
        source_dim = max(dimension * 2, n_initial)

        # Generate random vectors in higher dimension
        high_dim_vectors = np.random.randn(n_initial, source_dim)
        high_dim_vectors = normalize_vectors(high_dim_vectors)

        # Create random projection matrix
        projection_matrix = np.random.randn(source_dim, dimension) / np.sqrt(dimension)

        # Project to target dimension
        vectors = np.dot(high_dim_vectors, projection_matrix)
        vectors = normalize_vectors(vectors)

        # If we oversampled, prune to target count
        if n_initial > count:
            vectors = self._prune_vectors(vectors, count, verbose)

        return vectors

    def _prune_vectors(
        self, vectors: np.ndarray, target_count: int, verbose: bool
    ) -> np.ndarray:
        """Prune vectors using the same strategy as RandomSamplingGenerator."""
        generator = RandomSamplingGenerator(seed=self.seed, oversampling_factor=1.0)
        return generator._prune_vectors(vectors, target_count, verbose)


class EnergyMinimizationGenerator(BaseGenerator):
    """
    Generate vectors by minimizing energy function using gradient descent.

    This method optimizes vector positions to minimize an energy function
    based on pairwise distances, effectively spreading vectors apart.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        energy_power: float = 12.0,
        max_iterations: int = 300,
        learning_rate: float = 0.01,
        device: Optional[str] = None,
    ):
        """
        Initialize energy minimization generator.

        Args:
            seed: Random seed for reproducibility
            energy_power: Power in the energy function (higher = stronger repulsion)
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__(seed)
        self.energy_power = energy_power
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

    def generate(self, dimension: int, count: int, verbose: bool = False) -> np.ndarray:
        """Generate vectors using energy minimization."""
        if self.seed is not None:
            set_seed(self.seed)

        # Initialize random vectors on device
        vectors = torch.randn(count, dimension, device=self.device, requires_grad=True)
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        vectors = torch.nn.Parameter(vectors)

        # Setup optimizer
        optimizer = torch.optim.Adam([vectors], lr=self.learning_rate)

        # Optimization loop
        best_vectors = vectors.clone().detach()
        best_epsilon = float("inf")

        if verbose:
            pbar = tqdm(range(self.max_iterations), desc="Energy minimization")
        else:
            pbar = range(self.max_iterations)

        for step in pbar:
            optimizer.zero_grad()

            # Normalize vectors
            normalized = torch.nn.functional.normalize(vectors, p=2, dim=1)

            # Include antipodal points for better coverage
            augmented = torch.cat([normalized, -normalized], dim=0)

            # Compute pairwise distances
            cos_sim = torch.matmul(augmented, augmented.T)
            cos_sim.fill_diagonal_(0)

            # Energy function: 1/distance^power
            distances = torch.sqrt(2 - 2 * cos_sim + 1e-8)
            mask = torch.triu(torch.ones_like(distances), diagonal=1)
            energy = (mask * (1 / (distances**self.energy_power + 1e-8))).sum()

            # Track best result
            with torch.no_grad():
                cos_sim_vectors = torch.matmul(normalized, normalized.T)
                cos_sim_vectors.fill_diagonal_(0)
                current_epsilon = torch.max(torch.abs(cos_sim_vectors)).item()

                if current_epsilon < best_epsilon:
                    best_epsilon = current_epsilon
                    best_vectors = normalized.clone()

                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        {"ε": f"{current_epsilon:.4f}", "best_ε": f"{best_epsilon:.4f}"}
                    )

            # Backward pass
            energy.backward()
            optimizer.step()

        # Return best vectors found
        return best_vectors.cpu().numpy()


def create_generator(method: str, **kwargs) -> BaseGenerator:
    """
    Factory function to create generators.

    Args:
        method: One of 'random', 'jl', 'energy'
        **kwargs: Method-specific parameters

    Returns:
        Generator instance
    """
    if method == "random":
        return RandomSamplingGenerator(**kwargs)
    elif method == "jl":
        return JohnsonLindenstraussGenerator(**kwargs)
    elif method == "energy":
        return EnergyMinimizationGenerator(**kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from 'random', 'jl', 'energy'"
        )
