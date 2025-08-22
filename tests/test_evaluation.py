"""Tests for evaluation metrics."""

import pytest
import numpy as np
from src.evaluation import (
    evaluate_vectors,
    compare_methods,
    format_results_table
)


class TestEvaluateVectors:
    """Test vector evaluation."""
    
    def test_basic_evaluation(self):
        """Test basic evaluation of vectors."""
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        result = evaluate_vectors(vectors, generation_time=1.5, method_name="test")
        
        assert result['method'] == 'test'
        assert result['count'] == 3
        assert result['dimension'] == 3
        assert np.isclose(result['epsilon'], 0.0, atol=1e-10)
        assert result['generation_time'] == 1.5
    
    def test_empty_vectors(self):
        """Test evaluation of empty vector set."""
        vectors = np.array([])
        result = evaluate_vectors(vectors, generation_time=0.0, method_name="empty")
        
        assert result['count'] == 0
        assert result['dimension'] == 0
        assert np.isnan(result['epsilon'])
    
    def test_high_similarity_vectors(self):
        """Test vectors with high similarity."""
        vectors = np.array([
            [1, 0],
            [0.9, 0.1]
        ])
        
        result = evaluate_vectors(vectors, method_name="high_sim")
        
        assert result['count'] == 2
        assert result['dimension'] == 2
        # Should have high epsilon (close to 1)
        assert result['epsilon'] > 0.9


class TestCompareMethods:
    """Test method comparison."""
    
    def test_compare_multiple_methods(self):
        """Test comparing results from multiple methods."""
        results = [
            {'method': 'method1', 'epsilon': 0.5, 'generation_time': 1.0},
            {'method': 'method2', 'epsilon': 0.3, 'generation_time': 2.0},
            {'method': 'method3', 'epsilon': 0.7, 'generation_time': 0.5}
        ]
        
        comparison = compare_methods(results)
        
        assert comparison['best_method'] == 'method2'
        assert comparison['best_epsilon'] == 0.3
        assert len(comparison['all_results']) == 3
        assert 'method1' in comparison['summary']
        assert comparison['summary']['method1']['epsilon'] == 0.5
    
    def test_compare_empty_results(self):
        """Test comparing empty results."""
        comparison = compare_methods([])
        assert comparison == {}
    
    def test_compare_with_nan(self):
        """Test comparing with NaN values."""
        results = [
            {'method': 'method1', 'epsilon': float('nan'), 'generation_time': 1.0},
            {'method': 'method2', 'epsilon': 0.5, 'generation_time': 2.0}
        ]
        
        comparison = compare_methods(results)
        
        assert comparison['best_method'] == 'method2'
        assert comparison['best_epsilon'] == 0.5


class TestFormatResultsTable:
    """Test results table formatting."""
    
    def test_format_table(self):
        """Test formatting results as table."""
        results = [
            {'method': 'random', 'dimension': 8, 'count': 10, 'epsilon': 0.5},
            {'method': 'random', 'dimension': 8, 'count': 20, 'epsilon': 0.6},
            {'method': 'jl', 'dimension': 8, 'count': 10, 'epsilon': 0.4},
            {'method': 'jl', 'dimension': 8, 'count': 20, 'epsilon': 0.55},
            {'method': 'random', 'dimension': 16, 'count': 10, 'epsilon': 0.3},
            {'method': 'jl', 'dimension': 16, 'count': 10, 'epsilon': 0.25}
        ]
        
        table = format_results_table(results, dimensions=[8, 16], counts=[10, 20])
        
        # Check that table contains expected elements
        assert "Dimension 8" in table
        assert "Dimension 16" in table
        assert "random" in table
        assert "jl" in table
        assert "0.500000" in table  # epsilon value for random, d=8, n=10
        assert "0.400000" in table  # epsilon value for jl, d=8, n=10
    
    def test_format_empty_table(self):
        """Test formatting empty results."""
        table = format_results_table([], dimensions=[8], counts=[10])
        
        assert "RESULTS TABLE" in table
        assert "Maximum Absolute Cosine Similarity" in table
    
    def test_format_partial_results(self):
        """Test formatting with missing combinations."""
        results = [
            {'method': 'random', 'dimension': 8, 'count': 10, 'epsilon': 0.5}
        ]
        
        table = format_results_table(results, dimensions=[8, 16], counts=[10, 20])
        
        assert "Dimension 8" in table
        assert "random" in table
        assert "0.500000" in table
