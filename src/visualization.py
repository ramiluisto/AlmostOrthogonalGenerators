"""Visualization utilities for vector generation results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def plot_epsilon_vs_count(results: List[Dict[str, Any]], 
                          output_path: Optional[Path] = None,
                          title: Optional[str] = None) -> plt.Figure:
    """
    Plot epsilon (max abs cosine) vs vector count for different methods.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group results by method and dimension
    data = {}
    for result in results:
        method = result['method']
        dim = result['dimension']
        count = result['count']
        epsilon = result['epsilon']
        
        if not np.isfinite(epsilon):
            continue
            
        key = f"{method} (d={dim})"
        if key not in data:
            data[key] = {'counts': [], 'epsilons': []}
        data[key]['counts'].append(count)
        data[key]['epsilons'].append(epsilon)
    
    # Plot each method
    colors = sns.color_palette("husl", len(data))
    for (key, values), color in zip(data.items(), colors):
        # Sort by count
        sorted_pairs = sorted(zip(values['counts'], values['epsilons']))
        counts = [p[0] for p in sorted_pairs]
        epsilons = [p[1] for p in sorted_pairs]
        
        ax.plot(counts, epsilons, 'o-', label=key, color=color, markersize=6)
    
    ax.set_xlabel('Number of Vectors', fontsize=12)
    ax.set_ylabel('Maximum Absolute Cosine Similarity (ε)', fontsize=12)
    ax.set_title(title or 'Vector Generation Performance', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_dimension_comparison(results: List[Dict[str, Any]],
                             dimensions: List[int],
                             output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create subplots comparing methods across different dimensions.
    
    Args:
        results: List of evaluation results
        dimensions: List of dimensions to plot
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    sns.set_style("whitegrid")
    
    # Determine subplot layout
    n_dims = len(dimensions)
    cols = min(3, n_dims)
    rows = (n_dims + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Group results by dimension
    dim_data = {dim: [] for dim in dimensions}
    for result in results:
        if result['dimension'] in dim_data:
            dim_data[result['dimension']].append(result)
    
    # Plot each dimension
    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        dim_results = dim_data[dim]
        
        if not dim_results:
            ax.text(0.5, 0.5, f'No data for d={dim}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Dimension {dim}')
            continue
        
        # Group by method
        method_data = {}
        for result in dim_results:
            method = result['method']
            if method not in method_data:
                method_data[method] = {'counts': [], 'epsilons': []}
            if np.isfinite(result['epsilon']):
                method_data[method]['counts'].append(result['count'])
                method_data[method]['epsilons'].append(result['epsilon'])
        
        # Plot each method
        colors = sns.color_palette("deep", len(method_data))
        for (method, data), color in zip(method_data.items(), colors):
            if data['counts']:
                sorted_pairs = sorted(zip(data['counts'], data['epsilons']))
                counts = [p[0] for p in sorted_pairs]
                epsilons = [p[1] for p in sorted_pairs]
                ax.plot(counts, epsilons, 'o-', label=method, color=color)
        
        ax.set_xlabel('Vector Count')
        ax.set_ylabel('ε')
        ax.set_title(f'Dimension {dim}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    # Hide unused subplots
    for idx in range(len(dimensions), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Performance Across Dimensions', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_time_comparison(results: List[Dict[str, Any]],
                        output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot generation time comparison across methods.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare data
    method_times = {}
    method_epsilons = {}
    
    for result in results:
        method = result['method']
        time = result['generation_time']
        epsilon = result['epsilon']
        count = result['count']
        
        if not np.isfinite(epsilon):
            continue
            
        if method not in method_times:
            method_times[method] = []
            method_epsilons[method] = []
        
        method_times[method].append(time)
        method_epsilons[method].append(epsilon)
    
    # Box plot of times
    if method_times:
        times_data = []
        labels = []
        for method, times in method_times.items():
            times_data.append(times)
            labels.append(method)
        
        ax1.boxplot(times_data, labels=labels)
        ax1.set_ylabel('Generation Time (seconds)')
        ax1.set_title('Time Performance')
        ax1.grid(True, alpha=0.3)
        
        # Rotate labels if needed
        if len(labels) > 3:
            ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Scatter plot: time vs epsilon
    colors = sns.color_palette("deep", len(method_times))
    for (method, times), color in zip(method_times.items(), colors):
        epsilons = method_epsilons[method]
        ax2.scatter(times, epsilons, label=method, alpha=0.6, s=50, color=color)
    
    ax2.set_xlabel('Generation Time (seconds)')
    ax2.set_ylabel('ε')
    ax2.set_title('Time vs Quality Trade-off')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Analysis', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
    return fig
