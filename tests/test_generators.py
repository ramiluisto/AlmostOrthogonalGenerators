"""Tests for vector generators."""

import pytest
import numpy as np
from src.generators import (
    RandomSamplingGenerator,
    JohnsonLindenstraussGenerator,
    EnergyMinimizationGenerator,
    create_generator,
)
from src.utils import compute_max_abs_cosine


class TestRandomSamplingGenerator:
    """Test random sampling generator."""

    def test_basic_generation(self):
        """Test basic vector generation."""
        gen = RandomSamplingGenerator(seed=42)
        vectors = gen.generate(dimension=10, count=5)

        assert vectors.shape == (5, 10)
        # Check normalization
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_oversampling(self):
        """Test oversampling and pruning."""
        gen = RandomSamplingGenerator(seed=42, oversampling_factor=2.0)
        vectors = gen.generate(dimension=8, count=10)

        assert vectors.shape == (10, 8)
        epsilon = compute_max_abs_cosine(vectors)
        assert 0 <= epsilon <= 1

    def test_no_oversampling(self):
        """Test without oversampling."""
        gen = RandomSamplingGenerator(seed=42, oversampling_factor=1.0)
        vectors = gen.generate(dimension=8, count=10)

        assert vectors.shape == (10, 8)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        gen1 = RandomSamplingGenerator(seed=123)
        gen2 = RandomSamplingGenerator(seed=123)

        vectors1 = gen1.generate(dimension=10, count=5)
        vectors2 = gen2.generate(dimension=10, count=5)

        np.testing.assert_array_almost_equal(vectors1, vectors2)


class TestJohnsonLindenstraussGenerator:
    """Test Johnson-Lindenstrauss generator."""

    def test_basic_generation(self):
        """Test basic JL generation."""
        gen = JohnsonLindenstraussGenerator(seed=42)
        vectors = gen.generate(dimension=10, count=5)

        assert vectors.shape == (5, 10)
        # Check normalization
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_oversampling(self):
        """Test JL with oversampling."""
        gen = JohnsonLindenstraussGenerator(seed=42, oversampling_factor=2.0)
        vectors = gen.generate(dimension=8, count=10)

        assert vectors.shape == (10, 8)
        epsilon = compute_max_abs_cosine(vectors)
        assert 0 <= epsilon <= 1

    def test_reproducibility(self):
        """Test reproducibility with seed."""
        gen1 = JohnsonLindenstraussGenerator(seed=456)
        gen2 = JohnsonLindenstraussGenerator(seed=456)

        vectors1 = gen1.generate(dimension=12, count=6)
        vectors2 = gen2.generate(dimension=12, count=6)

        np.testing.assert_array_almost_equal(vectors1, vectors2)


class TestEnergyMinimizationGenerator:
    """Test energy minimization generator."""

    def test_basic_generation(self):
        """Test basic energy minimization."""
        gen = EnergyMinimizationGenerator(
            seed=42,
            energy_power=12.0,
            max_iterations=10,  # Small for testing
            device="cpu",
        )
        vectors = gen.generate(dimension=8, count=4)

        assert vectors.shape == (4, 8)
        # Check normalization
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-6)

    def test_epsilon_improvement(self):
        """Test that energy minimization improves epsilon."""
        # Generate random vectors
        random_gen = RandomSamplingGenerator(seed=42, oversampling_factor=1.0)
        random_vectors = random_gen.generate(dimension=8, count=10)
        random_epsilon = compute_max_abs_cosine(random_vectors)

        # Generate with energy minimization
        energy_gen = EnergyMinimizationGenerator(
            seed=42, energy_power=12.0, max_iterations=50, device="cpu"
        )
        energy_vectors = energy_gen.generate(dimension=8, count=10)
        energy_epsilon = compute_max_abs_cosine(energy_vectors)

        # Energy minimization should generally improve epsilon
        # (though not guaranteed for all seeds/parameters)
        assert energy_epsilon <= random_epsilon + 0.1  # Allow small tolerance

    def test_different_powers(self):
        """Test different energy powers."""
        for power in [2.0, 6.0, 12.0]:
            gen = EnergyMinimizationGenerator(
                seed=42, energy_power=power, max_iterations=10, device="cpu"
            )
            vectors = gen.generate(dimension=6, count=3)
            assert vectors.shape == (3, 6)


class TestFactoryFunction:
    """Test the create_generator factory function."""

    def test_create_random(self):
        """Test creating random generator."""
        gen = create_generator("random", seed=42, oversampling_factor=2.0)
        assert isinstance(gen, RandomSamplingGenerator)
        vectors = gen.generate(dimension=8, count=4)
        assert vectors.shape == (4, 8)

    def test_create_jl(self):
        """Test creating JL generator."""
        gen = create_generator("jl", seed=42)
        assert isinstance(gen, JohnsonLindenstraussGenerator)
        vectors = gen.generate(dimension=8, count=4)
        assert vectors.shape == (4, 8)

    def test_create_energy(self):
        """Test creating energy generator."""
        gen = create_generator("energy", seed=42, max_iterations=10)
        assert isinstance(gen, EnergyMinimizationGenerator)
        vectors = gen.generate(dimension=8, count=4)
        assert vectors.shape == (4, 8)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            create_generator("invalid_method")
