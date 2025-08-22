"""Evaluation metrics for vector sets."""

import numpy as np
from typing import Dict, Any, List, Tuple
import time
from .utils import normalize_vectors, compute_max_abs_cosine


def evaluate_vectors(vectors: np.ndarray, 
                    generation_time: float = 0.0,
                    method_name: str = "") -> Dict[str, Any]:
    """
    Evaluate a set of vectors.
    
    Args:
        vectors: Array of shape (n_vectors, dimension)
        generation_time: Time taken to generate vectors
        method_name: Name of the generation method
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(vectors) == 0:
        return {
            'method': method_name,
            'count': 0,
            'dimension': 0,
            'epsilon': float('nan'),
            'generation_time': generation_time
        }
    
    vectors = normalize_vectors(vectors)
    epsilon = compute_max_abs_cosine(vectors)
    
    return {
        'method': method_name,
        'count': len(vectors),
        'dimension': vectors.shape[1],
        'epsilon': epsilon,
        'generation_time': generation_time
    }


def compare_methods(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare results from multiple methods.
    
    Args:
        results: List of evaluation dictionaries
        
    Returns:
        Comparison summary
    """
    if not results:
        return {}
    
    # Find best epsilon
    valid_results = [r for r in results if np.isfinite(r.get('epsilon', float('nan')))]
    if not valid_results:
        return {'best_method': None, 'best_epsilon': float('nan')}
    
    best_result = min(valid_results, key=lambda x: x['epsilon'])
    
    comparison = {
        'best_method': best_result['method'],
        'best_epsilon': best_result['epsilon'],
        'all_results': results,
        'summary': {}
    }
    
    for result in results:
        method = result['method']
        comparison['summary'][method] = {
            'epsilon': result['epsilon'],
            'time': result['generation_time']
        }
    
    return comparison


def format_results_table(results: List[Dict[str, Any]], 
                        dimensions: List[int],
                        counts: List[int]) -> str:
    """
    Format results as a text table.
    
    Args:
        results: List of evaluation results
        dimensions: List of dimensions tested
        counts: List of vector counts tested
        
    Returns:
        Formatted table string
    """
    # Group results by dimension and count
    table = {}
    for result in results:
        dim = result['dimension']
        count = result['count']
        method = result['method']
        epsilon = result['epsilon']
        
        if dim not in table:
            table[dim] = {}
        if count not in table[dim]:
            table[dim][count] = {}
        table[dim][count][method] = epsilon
    
    # Build table string
    lines = []
    lines.append("=" * 80)
    lines.append("RESULTS TABLE: Maximum Absolute Cosine Similarity (ε)")
    lines.append("=" * 80)
    
    for dim in sorted(dimensions):
        if dim not in table:
            continue
            
        lines.append(f"\nDimension {dim}:")
        lines.append("-" * 40)
        
        # Header
        methods = set()
        for count_data in table[dim].values():
            methods.update(count_data.keys())
        methods = sorted(methods)
        
        header = "Vectors | " + " | ".join(f"{m:>10}" for m in methods)
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data rows
        for count in sorted(counts):
            if count not in table[dim]:
                continue
                
            row = f"{count:7} |"
            for method in methods:
                if method in table[dim][count]:
                    epsilon = table[dim][count][method]
                    if np.isfinite(epsilon):
                        row += f" {epsilon:10.6f} |"
                    else:
                        row += f" {'N/A':>10} |"
                else:
                    row += f" {'-':>10} |"
            lines.append(row)
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
