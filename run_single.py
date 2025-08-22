#!/usr/bin/env python3
"""
Run a single vector generation experiment.

Examples:
    python run_single.py --method energy --dimension 32 --count 50
    python run_single.py --method jl --dimension 64 --count 100 --seed 42
    python run_single.py --method random --dimension 16 --count 30 --save-vectors
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from src.generators import create_generator
from src.evaluation import evaluate_vectors
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Generate almost orthogonal vectors using a single method')
    
    # Required arguments
    parser.add_argument('--method', type=str, required=True, 
                       choices=['random', 'jl', 'energy'],
                       help='Generation method to use')
    parser.add_argument('--dimension', type=int, required=True,
                       help='Vector dimension')
    parser.add_argument('--count', type=int, required=True,
                       help='Number of vectors to generate')
    
    # Optional arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                       help='Output directory (default: results)')
    parser.add_argument('--save-vectors', action='store_true',
                       help='Save generated vectors to file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show progress information')
    
    # Method-specific arguments
    parser.add_argument('--oversampling', type=float, default=2.0,
                       help='Oversampling factor for random/jl methods (default: 2.0)')
    parser.add_argument('--energy-power', type=float, default=12.0,
                       help='Energy power for energy method (default: 12.0)')
    parser.add_argument('--max-iterations', type=int, default=300,
                       help='Max iterations for energy method (default: 300)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for energy method (default: 0.01)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device for energy method (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create generator based on method
    print(f"\n{'='*60}")
    print(f"Running {args.method.upper()} method")
    print(f"Dimension: {args.dimension}, Target vectors: {args.count}")
    print(f"{'='*60}\n")
    
    if args.method == 'random':
        generator = create_generator(
            'random',
            seed=args.seed,
            oversampling_factor=args.oversampling
        )
    elif args.method == 'jl':
        generator = create_generator(
            'jl',
            seed=args.seed,
            oversampling_factor=args.oversampling
        )
    elif args.method == 'energy':
        generator = create_generator(
            'energy',
            seed=args.seed,
            energy_power=args.energy_power,
            max_iterations=args.max_iterations,
            learning_rate=args.learning_rate,
            device=args.device
        )
    
    # Generate vectors
    print(f"Generating vectors...")
    start_time = time.time()
    vectors = generator.generate(args.dimension, args.count, verbose=args.verbose)
    generation_time = time.time() - start_time
    
    # Evaluate results
    results = evaluate_vectors(vectors, generation_time, args.method)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Method: {results['method']}")
    print(f"Dimension: {results['dimension']}")
    print(f"Vectors generated: {results['count']}")
    print(f"Max absolute cosine similarity (ε): {results['epsilon']:.6f}")
    print(f"Generation time: {results['generation_time']:.3f} seconds")
    print(f"{'='*60}\n")
    
    # Save results
    results_file = args.output_dir / f"{args.method}_d{args.dimension}_n{args.count}_seed{args.seed}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Save vectors if requested
    if args.save_vectors:
        vectors_file = args.output_dir / f"{args.method}_d{args.dimension}_n{args.count}_seed{args.seed}_vectors.npy"
        np.save(vectors_file, vectors)
        print(f"Vectors saved to: {vectors_file}")


if __name__ == '__main__':
    main()
