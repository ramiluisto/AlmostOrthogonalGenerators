"""Tests for utility functions."""

import pytest
import numpy as np
import torch
from src.utils import (
    get_device,
    set_seed,
    normalize_vectors,
    compute_max_abs_cosine
)


class TestDeviceDetection:
    """Test device detection functionality."""
    
    def test_get_device_cpu_forced(self):
        """Test forcing CPU usage."""
        device = get_device(force_cpu=True)
        assert device.type == 'cpu'
    
    def test_get_device_auto(self):
        """Test automatic device detection."""
        device = get_device(force_cpu=False)
        # Should be either 'cpu' or 'cuda' depending on availability
        assert device.type in ['cpu', 'cuda']


class TestSeedSetting:
    """Test seed setting for reproducibility."""
    
    def test_numpy_seed(self):
        """Test that numpy seed is set correctly."""
        set_seed(123)
        arr1 = np.random.randn(5, 5)
        
        set_seed(123)
        arr2 = np.random.randn(5, 5)
        
        np.testing.assert_array_equal(arr1, arr2)
    
    def test_torch_seed(self):
        """Test that torch seed is set correctly."""
        set_seed(456)
        tensor1 = torch.randn(3, 3)
        
        set_seed(456)
        tensor2 = torch.randn(3, 3)
        
        torch.testing.assert_close(tensor1, tensor2)


class TestNormalization:
    """Test vector normalization."""
    
    def test_normalize_numpy(self):
        """Test normalization of numpy arrays."""
        vectors = np.random.randn(10, 5)
        normalized = normalize_vectors(vectors)
        
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
    
    def test_normalize_torch(self):
        """Test normalization of torch tensors."""
        vectors = torch.randn(10, 5)
        normalized = normalize_vectors(vectors)
        
        norms = torch.norm(normalized, dim=1)
        torch.testing.assert_close(norms, torch.ones(10), rtol=1e-6, atol=1e-6)
    
    def test_normalize_zero_vector(self):
        """Test handling of zero vectors."""
        vectors = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        normalized = normalize_vectors(vectors)
        
        # First and third vectors should be normalized
        assert np.allclose(np.linalg.norm(normalized[0]), 1.0)
        assert np.allclose(np.linalg.norm(normalized[2]), 1.0)
        
        # Zero vector should remain zero
        assert np.allclose(normalized[1], 0)


class TestCosineComputation:
    """Test cosine similarity computation."""
    
    def test_orthogonal_vectors(self):
        """Test with orthogonal vectors."""
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        epsilon = compute_max_abs_cosine(vectors)
        assert np.isclose(epsilon, 0.0, atol=1e-10)
    
    def test_parallel_vectors(self):
        """Test with parallel vectors."""
        vectors = np.array([
            [1, 0, 0],
            [1, 0, 0]
        ])
        epsilon = compute_max_abs_cosine(vectors)
        assert np.isclose(epsilon, 1.0, atol=1e-10)
    
    def test_antipodal_vectors(self):
        """Test with antipodal vectors."""
        vectors = np.array([
            [1, 0, 0],
            [-1, 0, 0]
        ])
        epsilon = compute_max_abs_cosine(vectors)
        assert np.isclose(epsilon, 1.0, atol=1e-10)
    
    def test_single_vector(self):
        """Test with single vector."""
        vectors = np.array([[1, 2, 3]])
        epsilon = compute_max_abs_cosine(vectors)
        assert epsilon == 0.0
    
    def test_random_vectors(self):
        """Test with random vectors."""
        np.random.seed(42)
        vectors = np.random.randn(20, 10)
        epsilon = compute_max_abs_cosine(vectors)
        
        # Epsilon should be between 0 and 1
        assert 0 <= epsilon <= 1
        
        # Manually compute to verify
        normalized = normalize_vectors(vectors)
        cos_sim = np.abs(np.dot(normalized, normalized.T))
        np.fill_diagonal(cos_sim, 0)
        expected_epsilon = np.max(cos_sim)
        
        assert np.isclose(epsilon, expected_epsilon)
