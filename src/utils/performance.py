"""
Performance profiling utilities for SG-MRM project.

Provides tools for measuring and analyzing computation performance.
"""

import time
import functools
from typing import Callable, Any, Dict
from datetime import datetime
import numpy as np


class PerformanceTimer:
    """
    Context manager for timing code blocks.

    Usage:
        with PerformanceTimer("My operation") as timer:
            # ... code to time ...
            pass
        print(f"Elapsed: {timer.elapsed_ms} ms")
    """

    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize performance timer.

        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing information
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and optionally print result."""
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

        if self.verbose:
            print(f"{self.name}: {self.elapsed_ms:.2f} ms")

        return False


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution.

    Usage:
        @timeit
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        print(f"{func.__name__}: {elapsed_ms:.2f} ms")
        return result
    return wrapper


class PerformanceProfiler:
    """
    Profiler for tracking performance across multiple operations.

    Collects timing statistics for different operations and provides
    summary reports.
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.timings: Dict[str, list] = {}
        self.current_operation = None
        self.operation_start = None

    def start(self, operation_name: str):
        """
        Start timing an operation.

        Args:
            operation_name: Name of the operation
        """
        self.current_operation = operation_name
        self.operation_start = time.perf_counter()

    def end(self):
        """End timing the current operation."""
        if self.current_operation is None:
            return

        elapsed = time.perf_counter() - self.operation_start
        elapsed_ms = elapsed * 1000

        if self.current_operation not in self.timings:
            self.timings[self.current_operation] = []

        self.timings[self.current_operation].append(elapsed_ms)
        self.current_operation = None
        self.operation_start = None

    def record(self, operation_name: str, elapsed_ms: float):
        """
        Manually record a timing.

        Args:
            operation_name: Name of the operation
            elapsed_ms: Elapsed time in milliseconds
        """
        if operation_name not in self.timings:
            self.timings[operation_name] = []
        self.timings[operation_name].append(elapsed_ms)

    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Dictionary with min, max, mean, median, std statistics
        """
        if operation_name not in self.timings:
            return {}

        times = np.array(self.timings[operation_name])

        return {
            'count': len(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'std_ms': np.std(times),
            'total_ms': np.sum(times)
        }

    def print_summary(self):
        """Print performance summary for all operations."""
        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)

        for operation_name in sorted(self.timings.keys()):
            stats = self.get_statistics(operation_name)
            print(f"\n{operation_name}:")
            print(f"  Count:  {stats['count']}")
            print(f"  Mean:   {stats['mean_ms']:.2f} ms")
            print(f"  Median: {stats['median_ms']:.2f} ms")
            print(f"  Min:    {stats['min_ms']:.2f} ms")
            print(f"  Max:    {stats['max_ms']:.2f} ms")
            print(f"  Std:    {stats['std_ms']:.2f} ms")
            print(f"  Total:  {stats['total_ms']:.2f} ms")

        print("=" * 70 + "\n")

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.current_operation = None
        self.operation_start = None


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """
    Get the global profiler instance.

    Returns:
        Global PerformanceProfiler instance
    """
    return _global_profiler


def profile_layer_computation(layer_name: str):
    """
    Decorator for profiling layer computation methods.

    Usage:
        @profile_layer_computation("L1")
        def compute(self, timestamp):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            profiler.start(f"{layer_name}_compute")
            result = func(*args, **kwargs)
            profiler.end()
            return result
        return wrapper
    return decorator
