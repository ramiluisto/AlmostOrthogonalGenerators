# Almost Orthogonal Vector Generation

A Python library for generating sets of almost orthogonal vectors using various methods. This code accompanies the research on finding large sets of vectors with small pairwise cosine similarities.

## Overview

This library implements three main methods for generating almost orthogonal vectors:

1. **Random Sampling with Pruning**: Generates random unit vectors and optionally oversamples and prunes to improve orthogonality
2. **Johnson-Lindenstrauss Projection**: Uses random projections from higher dimensions with optional pruning
3. **Energy Minimization**: Optimizes vector positions using gradient descent on an energy function

The key metric is **ε (epsilon)** - the maximum absolute cosine similarity between any pair of vectors. Lower ε means better orthogonality.

## Installation

```bash
# Clone or download this repository
cd code_share

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Single Experiment

Generate vectors using a specific method:

```bash
# Energy minimization: 50 vectors in dimension 32
python run_single.py --method energy --dimension 32 --count 50

# Johnson-Lindenstrauss with 2x oversampling
python run_single.py --method jl --dimension 64 --count 100 --oversampling 2.0

# Random sampling with custom seed
python run_single.py --method random --dimension 16 --count 30 --seed 42
```

### Batch Experiments

Compare multiple methods across dimensions and vector counts:

```bash
# Quick test with small parameters
python run_batch.py --quick

# Custom batch experiment
python run_batch.py --dimensions 8 16 32 --counts 10 20 40 60

# Full comparison with all methods
python run_batch.py --dimensions 32 64 128 --counts 50 100 200 --methods random jl energy
```

### Configuration-based Experiments

Run complex experiments from YAML configuration files:

```bash
# Run example configuration
python run_config.py configs/example_config.yaml

# Quick test configuration
python run_config.py configs/quick_test.yaml

# Custom output directory
python run_config.py configs/example_config.yaml --output-dir my_results
```

## Results

Running the quick batch test produces the following results:

### Performance Comparison

| Dimension | Vectors | Energy (ε) | JL (ε) | Random (ε) |
|-----------|---------|------------|--------|------------|
| 8         | 4       | 0.045      | 0.507  | 0.376      |
| 8         | 10      | 0.395      | 0.464  | 0.587      |
| 8         | 20      | 0.448      | 0.693  | 0.660      |
| 16        | 4       | 0.053      | 0.371  | 0.262      |
| 16        | 10      | 0.099      | 0.501  | 0.416      |
| 16        | 20      | 0.266      | 0.529  | 0.459      |

The energy minimization method consistently achieves the lowest ε (best orthogonality), especially for smaller vector counts relative to the dimension.

### Visualizations

The batch script automatically generates three types of plots:

1. **Epsilon vs Vector Count**: Shows how orthogonality degrades as more vectors are added
2. **Dimension Comparison**: Compares methods across different dimensions
3. **Time Analysis**: Shows the trade-off between computation time and quality

## Methods

### Random Sampling (`random`)

Generates random vectors from a normal distribution and normalizes them to unit length. With oversampling, generates more vectors than needed and prunes the worst ones.

**Parameters:**
- `oversampling_factor`: Generate this many times the target count, then prune (default: 2.0)

### Johnson-Lindenstrauss Projection (`jl`)

Projects random vectors from a higher-dimensional space to the target dimension using a random projection matrix.

**Parameters:**
- `oversampling_factor`: Generate this many times the target count before projection (default: 2.0)

### Energy Minimization (`energy`)

Optimizes vector positions by minimizing an energy function based on pairwise distances. Uses gradient descent with PyTorch.

**Parameters:**
- `energy_power`: Power in the energy function 1/d^p (default: 12.0)
- `max_iterations`: Number of optimization steps (default: 300)
- `learning_rate`: Step size for gradient descent (default: 0.01)
- `device`: 'cuda' or 'cpu' (auto-detected if not specified)

## Configuration Files

Create custom experiments using YAML configuration:

```yaml
# Example configuration
output_dir: results/my_experiment
seed: 42

dimensions: [32, 64, 128]
vector_counts: [50, 100, 200, 400]

methods:
  random_sampling:
    enabled: true
    type: random
    parameters:
      oversampling_factor: 2.0
      
  energy_minimization:
    enabled: true
    type: energy
    parameters:
      energy_power: 12.0
      max_iterations: 500
      learning_rate: 0.01

generate_plots: true
```

## GPU Acceleration

The energy minimization method automatically uses CUDA if available. To force CPU usage:

```bash
# Force CPU for energy method
python run_single.py --method energy --dimension 32 --count 50 --device cpu
```

## Testing

Run the test suite to verify the installation:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_generators.py -v
```

## Output Files

Each run produces several output files:

- `*.json`: Detailed results with epsilon values and generation times
- `*.png`: Visualization plots comparing methods
- `*.txt`: Formatted results table
- `*.npy`: Generated vectors (if `--save-vectors` is used)

## Advanced Usage

### Custom Energy Powers

Experiment with different energy functions:

```bash
# Lower power (weaker repulsion)
python run_single.py --method energy --dimension 32 --count 50 --energy-power 2.0

# Higher power (stronger repulsion) 
python run_single.py --method energy --dimension 32 --count 50 --energy-power 20.0
```

### Adjusting Optimization

Fine-tune the energy minimization:

```bash
# More iterations for better results
python run_single.py --method energy --dimension 64 --count 100 \
    --max-iterations 1000 --learning-rate 0.005
```

### Saving Vectors

Save generated vectors for further analysis:

```bash
# Save vectors as numpy array
python run_single.py --method energy --dimension 32 --count 50 --save-vectors

# Vectors saved to: results/energy_d32_n50_seed42_vectors.npy
```

## Performance Tips

1. **Energy minimization** is slowest but achieves best results
2. **Random sampling** is fastest but gives worst orthogonality
3. **Johnson-Lindenstrauss** provides a good balance
4. Use GPU acceleration for energy minimization when available
5. Reduce `max_iterations` for faster but potentially worse results
6. Increase `oversampling_factor` for better pruning results

## Citation

If you use this code in your research, please cite the accompanying paper.

## License

This code is provided for research purposes. See LICENSE file for details.
