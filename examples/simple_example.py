#!/usr/bin/env python3
"""
Simple example demonstrating the three vector generation methods.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators import create_generator
from src.evaluation import evaluate_vectors
from src.utils import set_seed


def main():
    # Set seed for reproducibility
    set_seed(42)

    # Test parameters
    dimension = 16
    count = 20

    print(f"Generating {count} vectors in dimension {dimension}")
    print("=" * 60)

    # Method 1: Random Sampling
    print("\n1. Random Sampling with 2x oversampling:")
    random_gen = create_generator("random", seed=42, oversampling_factor=2.0)
    random_vectors = random_gen.generate(dimension, count)
    random_results = evaluate_vectors(random_vectors, method_name="Random")
    print(f"   ε = {random_results['epsilon']:.6f}")

    # Method 2: Johnson-Lindenstrauss
    print("\n2. Johnson-Lindenstrauss projection:")
    jl_gen = create_generator("jl", seed=42, oversampling_factor=2.0)
    jl_vectors = jl_gen.generate(dimension, count)
    jl_results = evaluate_vectors(jl_vectors, method_name="JL")
    print(f"   ε = {jl_results['epsilon']:.6f}")

    # Method 3: Energy Minimization
    print("\n3. Energy Minimization (100 iterations):")
    energy_gen = create_generator(
        "energy", seed=42, energy_power=12.0, max_iterations=100
    )
    energy_vectors = energy_gen.generate(dimension, count, verbose=True)
    energy_results = evaluate_vectors(energy_vectors, method_name="Energy")
    print(f"   ε = {energy_results['epsilon']:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Random:  ε = {random_results['epsilon']:.6f}")
    print(f"JL:      ε = {jl_results['epsilon']:.6f}")
    print(f"Energy:  ε = {energy_results['epsilon']:.6f}")
    print("=" * 60)

    # Best method
    methods = [random_results, jl_results, energy_results]
    best = min(methods, key=lambda x: x["epsilon"])
    print(f"\nBest method: {best['method']} with ε = {best['epsilon']:.6f}")


if __name__ == "__main__":
    main()
