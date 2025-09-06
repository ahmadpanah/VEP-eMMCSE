"""
Tests for benchmarking utilities
"""

import pytest
import tempfile
from vep_emmcse.utils.benchmark import (
    VEP_eMMCSE_Benchmark, BenchmarkResult, ComparisonResult
)
from vep_emmcse.schemes.vep_emmcse import create_vep_emmcse
from vep_emmcse.schemes.emmcse_baseline import create_emmcse_pq

@pytest.fixture
def benchmark():
    """Create benchmark instance with test configuration"""
    scheme = create_vep_emmcse()
    return VEP_eMMCSE_Benchmark(scheme)

def test_measure_operation(benchmark):
    """Test operation measurement"""
    def test_op():
        # Simple operation for testing
        result = 0
        for i in range(1000):
            result += i
        return result
    
    result = benchmark.measure_operation(
        operation_name="test_operation",
        operation_func=test_op,
        num_runs=10,
        progress_bar=False
    )
    
    assert isinstance(result, BenchmarkResult)
    assert result.operation_name == "test_operation"
    assert result.mean_time_ms > 0
    assert result.memory_usage_mb >= 0

def test_crypto_primitives_benchmark(benchmark):
    """Test cryptographic primitives benchmarking"""
    results = benchmark.benchmark_crypto_primitives()
    assert len(results) >= 3  # At least keygen, sign, verify
    
    for name, result in results.items():
        assert isinstance(result, BenchmarkResult)
        assert result.num_runs > 0
        assert result.mean_time_ms > 0

def test_update_performance_benchmark(benchmark):
    """Test update operation benchmarking"""
    results = benchmark.benchmark_update_performance()
    assert len(results) > 0
    
    # Check scaling with keyword count
    times = []
    for result in results.values():
        assert isinstance(result, BenchmarkResult)
        times.append(result.mean_time_ms)
    
    # Verify time increases with keyword count
    assert all(times[i] <= times[i+1] for i in range(len(times)-1))

def test_search_performance_benchmark(benchmark):
    """Test search operation benchmarking"""
    results = benchmark.benchmark_search_performance()
    assert len(results) > 0
    
    # Check both token generation and search results
    token_results = [r for r in results.values() if "Token Gen" in r.operation_name]
    search_results = [r for r in results.values() if "Search" in r.operation_name]
    
    assert len(token_results) > 0
    assert len(search_results) > 0

def test_verifiability_overhead_benchmark(benchmark):
    """Test verification overhead benchmarking"""
    results = benchmark.benchmark_verifiability_overhead()
    assert len(results) > 0
    
    # Check scaling with result set size
    for result in results.values():
        assert isinstance(result, BenchmarkResult)
        assert "Verify" in result.operation_name

def test_comparison_with_baseline(benchmark):
    """Test comparison with baseline scheme"""
    comparison_results = benchmark.run_comparison_with_baseline()
    assert len(comparison_results) > 0
    
    for comp in comparison_results:
        assert isinstance(comp, ComparisonResult)
        assert comp.vep_emmcse_time_ms > 0
        assert comp.baseline_time_ms > 0
        assert comp.overhead_factor > 0

def test_report_generation(benchmark):
    """Test performance report generation"""
    # Run some benchmarks first
    benchmark.benchmark_crypto_primitives()
    
    with tempfile.NamedTemporaryFile(suffix='.html') as tmp:
        benchmark.generate_performance_report(tmp.name)
        
        with open(tmp.name, 'r') as f:
            content = f.read()
            assert '<html>' in content
            assert 'VEP-eMMCSE Performance' in content
            assert '<table>' in content

def test_full_benchmark_suite(benchmark):
    """Test running complete benchmark suite"""
    results = benchmark.run_all_benchmarks()
    
    assert 'crypto' in results
    assert 'update' in results
    assert 'search' in results
    assert 'verify' in results
    assert 'comparison' in results

def test_export_results(benchmark):
    """Test results export functionality"""
    # Run some benchmarks
    benchmark.benchmark_crypto_primitives()
    
    with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
        benchmark.export_results_json(tmp.name)
        
        # Verify file exists and contains valid JSON
        import json
        with open(tmp.name, 'r') as f:
            data = json.load(f)
            assert 'benchmark_results' in data
            assert 'metadata' in data
