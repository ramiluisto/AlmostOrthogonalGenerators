#!/usr/bin/env python3
"""
Run batch experiments comparing multiple methods across dimensions and vector counts.

Examples:
    python run_batch.py --dimensions 8 16 32 --counts 10 20 40
    python run_batch.py --dimensions 64 128 --counts 50 100 200 --methods random jl
    python run_batch.py --quick  # Run a quick test with small values
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.generators import create_generator
from src.evaluation import evaluate_vectors, compare_methods, format_results_table
from src.visualization import plot_epsilon_vs_count, plot_dimension_comparison, plot_time_comparison
from src.utils import set_seed


def run_experiment(method: str, dimension: int, count: int, 
                  seed: int = 42, verbose: bool = False,
                  **method_kwargs) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    
    # Create generator
    if method == 'random':
        generator = create_generator('random', seed=seed, 
                                    oversampling_factor=method_kwargs.get('oversampling', 2.0))
    elif method == 'jl':
        generator = create_generator('jl', seed=seed,
                                    oversampling_factor=method_kwargs.get('oversampling', 2.0))
    elif method == 'energy':
        generator = create_generator('energy', seed=seed,
                                    energy_power=method_kwargs.get('energy_power', 12.0),
                                    max_iterations=method_kwargs.get('max_iterations', 300),
                                    learning_rate=method_kwargs.get('learning_rate', 0.01),
                                    device=method_kwargs.get('device'))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Generate vectors
    start_time = time.time()
    vectors = generator.generate(dimension, count, verbose=verbose)
    generation_time = time.time() - start_time
    
    # Evaluate
    results = evaluate_vectors(vectors, generation_time, method)
    return results


def main():
    parser = argparse.ArgumentParser(description='Run batch vector generation experiments')
    
    # Experiment parameters
    parser.add_argument('--dimensions', type=int, nargs='+', 
                       default=[8, 16, 32],
                       help='List of dimensions to test')
    parser.add_argument('--counts', type=int, nargs='+',
                       default=[4, 10, 20, 40, 60, 100],
                       help='List of vector counts to test')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['random', 'jl', 'energy'],
                       choices=['random', 'jl', 'energy'],
                       help='Methods to compare')
    
    # Quick test option
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with small values')
    
    # Output options
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Show progress for each generation')
    
    # Method-specific parameters
    parser.add_argument('--oversampling', type=float, default=2.0,
                       help='Oversampling factor for random/jl methods')
    parser.add_argument('--energy-power', type=float, default=12.0,
                       help='Energy power for energy method')
    parser.add_argument('--max-iterations', type=int, default=300,
                       help='Max iterations for energy method')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device for energy method')
    
    args = parser.parse_args()
    
    # Override for quick test
    if args.quick:
        args.dimensions = [8, 16]
        args.counts = [4, 10, 20]
        args.max_iterations = 50
        print("Running quick test with reduced parameters...")
    
    # Set global seed
    set_seed(args.seed)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print experiment summary
    print(f"\n{'='*60}")
    print("BATCH EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Vector counts: {args.counts}")
    print(f"Methods: {args.methods}")
    print(f"Total experiments: {len(args.dimensions) * len(args.counts) * len(args.methods)}")
    print(f"{'='*60}\n")
    
    # Run experiments
    all_results = []
    experiment_count = 0
    total_experiments = len(args.dimensions) * len(args.counts) * len(args.methods)
    
    for dimension in args.dimensions:
        for count in args.counts:
            for method in args.methods:
                experiment_count += 1
                print(f"[{experiment_count}/{total_experiments}] "
                      f"Running {method} for d={dimension}, n={count}...")
                
                try:
                    result = run_experiment(
                        method, dimension, count,
                        seed=args.seed,
                        verbose=args.verbose,
                        oversampling=args.oversampling,
                        energy_power=args.energy_power,
                        max_iterations=args.max_iterations,
                        learning_rate=0.01,
                        device=args.device
                    )
                    all_results.append(result)
                    print(f"  ✓ ε = {result['epsilon']:.6f}, time = {result['generation_time']:.2f}s")
                    
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    all_results.append({
                        'method': method,
                        'dimension': dimension,
                        'count': count,
                        'epsilon': float('nan'),
                        'generation_time': 0.0,
                        'error': str(e)
                    })
    
    # Save raw results
    results_file = args.output_dir / 'batch_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate comparison summary
    comparison = compare_methods(all_results)
    comparison_file = args.output_dir / 'comparison_summary.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print results table
    table = format_results_table(all_results, args.dimensions, args.counts)
    print(table)
    
    # Save table to file
    table_file = args.output_dir / 'results_table.txt'
    with open(table_file, 'w') as f:
        f.write(table)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Main comparison plot
    fig1 = plot_epsilon_vs_count(
        all_results,
        output_path=args.output_dir / 'epsilon_vs_count.png',
        title='Method Comparison: ε vs Vector Count'
    )
    
    # Dimension-specific plots
    fig2 = plot_dimension_comparison(
        all_results,
        args.dimensions,
        output_path=args.output_dir / 'dimension_comparison.png'
    )
    
    # Time analysis
    fig3 = plot_time_comparison(
        all_results,
        output_path=args.output_dir / 'time_analysis.png'
    )
    
    print(f"Plots saved to: {args.output_dir}/")
    
    # Print summary
    if comparison and 'best_method' in comparison:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Best method: {comparison['best_method']}")
        print(f"Best ε achieved: {comparison['best_epsilon']:.6f}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
