#!/usr/bin/env python3
"""
Run experiments from a YAML configuration file.

Example:
    python run_config.py configs/example_config.yaml
    python run_config.py configs/large_experiment.yaml --output-dir custom_results
"""

import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from src.generators import create_generator
from src.evaluation import evaluate_vectors, format_results_table
from src.visualization import (
    plot_epsilon_vs_count,
    plot_dimension_comparison,
    plot_time_comparison,
)
from src.utils import set_seed


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config(
    config: Dict[str, Any], output_dir: Path, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Run experiments based on configuration."""

    # Extract parameters
    dimensions = config.get("dimensions", [32])
    vector_counts = config.get("vector_counts", [50])
    methods_config = config.get("methods", {})
    global_seed = config.get("seed", 42)

    # Set global seed
    set_seed(global_seed)

    # Prepare results
    all_results = []
    total_experiments = len(dimensions) * len(vector_counts) * len(methods_config)
    experiment_count = 0

    print(f"\n{'='*60}")
    print(f"Running experiments from config")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*60}\n")

    # Run experiments
    for dimension in dimensions:
        for count in vector_counts:
            for method_name, method_params in methods_config.items():
                experiment_count += 1

                # Skip if method is disabled
                if not method_params.get("enabled", True):
                    continue

                print(
                    f"[{experiment_count}/{total_experiments}] "
                    f"{method_name}: d={dimension}, n={count}"
                )

                try:
                    # Get method type and parameters
                    method_type = method_params.get("type", method_name)
                    params = method_params.get("parameters", {})

                    # Override seed if specified
                    seed = params.get("seed", global_seed)

                    # Create generator
                    if method_type in ["random", "random_sampling"]:
                        generator = create_generator(
                            "random",
                            seed=seed,
                            oversampling_factor=params.get("oversampling_factor", 2.0),
                        )
                    elif method_type in ["jl", "johnson_lindenstrauss"]:
                        generator = create_generator(
                            "jl",
                            seed=seed,
                            oversampling_factor=params.get("oversampling_factor", 2.0),
                        )
                    elif method_type in ["energy", "energy_minimization"]:
                        generator = create_generator(
                            "energy",
                            seed=seed,
                            energy_power=params.get("energy_power", 12.0),
                            max_iterations=params.get("max_iterations", 300),
                            learning_rate=params.get("learning_rate", 0.01),
                            device=params.get("device"),
                        )
                    else:
                        print(f"  ✗ Unknown method type: {method_type}")
                        continue

                    # Generate vectors
                    start_time = time.time()
                    vectors = generator.generate(dimension, count, verbose=verbose)
                    generation_time = time.time() - start_time

                    # Evaluate
                    result = evaluate_vectors(vectors, generation_time, method_name)
                    all_results.append(result)

                    print(
                        f"  ✓ ε = {result['epsilon']:.6f}, time = {result['generation_time']:.2f}s"
                    )

                    # Save vectors if requested
                    if params.get("save_vectors", False):
                        vectors_dir = output_dir / "vectors"
                        vectors_dir.mkdir(exist_ok=True)
                        vectors_file = (
                            vectors_dir / f"{method_name}_d{dimension}_n{count}.npy"
                        )
                        np.save(vectors_file, vectors)

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    all_results.append(
                        {
                            "method": method_name,
                            "dimension": dimension,
                            "count": count,
                            "epsilon": float("nan"),
                            "generation_time": 0.0,
                            "error": str(e),
                        }
                    )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments from configuration file"
    )
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (overrides config)"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.get("output_dir", "results"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output directory for reference
    config_copy = output_dir / "experiment_config.yaml"
    with open(config_copy, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run experiments
    results = run_from_config(config, output_dir, args.verbose)

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate table
    dimensions = config.get("dimensions", [])
    counts = config.get("vector_counts", [])
    table = format_results_table(results, dimensions, counts)
    print(table)

    table_file = output_dir / "results_table.txt"
    with open(table_file, "w") as f:
        f.write(table)

    # Generate plots if requested
    if config.get("generate_plots", True):
        print("\nGenerating visualizations...")

        plot_epsilon_vs_count(
            results,
            output_path=output_dir / "epsilon_vs_count.png",
            title=config.get("plot_title", "Vector Generation Comparison"),
        )

        plot_dimension_comparison(
            results, dimensions, output_path=output_dir / "dimension_comparison.png"
        )

        plot_time_comparison(results, output_path=output_dir / "time_analysis.png")

        print(f"Plots saved to: {output_dir}/")

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()
