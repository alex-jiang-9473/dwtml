"""
Resource monitoring utilities for GPU memory tracking and logging.
Provides functions to monitor CUDA memory usage during training.
"""

import torch
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os


@dataclass
class MemoryStats:
    """Container for GPU memory statistics."""
    peak_reserved_mb: float
    peak_allocated_mb: float
    current_reserved_mb: float
    current_allocated_mb: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'peak_reserved_mb': round(self.peak_reserved_mb, 2),
            'peak_allocated_mb': round(self.peak_allocated_mb, 2),
            'current_reserved_mb': round(self.current_reserved_mb, 2),
            'current_allocated_mb': round(self.current_allocated_mb, 2),
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation of memory stats."""
        return (
            f"Memory Stats - "
            f"Peak Reserved: {self.peak_reserved_mb:.2f} MB, "
            f"Peak Allocated: {self.peak_allocated_mb:.2f} MB, "
            f"Current Reserved: {self.current_reserved_mb:.2f} MB, "
            f"Current Allocated: {self.current_allocated_mb:.2f} MB"
        )


class ResourceMonitor:
    """Monitor GPU memory and training resource usage."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize resource monitor.
        
        Args:
            device: Device to monitor ('cuda' or 'cpu')
        """
        self.device = device
        self.is_cuda = device == 'cuda' and torch.cuda.is_available()
        self.start_time = None
        self.memory_history = []
        self.iteration_times = []
        
    def start(self):
        """Start monitoring session."""
        self.start_time = time.time()
        self.memory_history = []
        self.iteration_times = []
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """
        Get current memory statistics.
        
        Returns:
            MemoryStats object or None if not on CUDA
        """
        if not self.is_cuda:
            return None
        
        # Convert bytes to MB
        peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
        peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        current_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        
        stats = MemoryStats(
            peak_reserved_mb=peak_reserved,
            peak_allocated_mb=peak_allocated,
            current_reserved_mb=current_reserved,
            current_allocated_mb=current_allocated,
            timestamp=time.time()
        )
        self.memory_history.append(stats)
        return stats
    
    def log_iteration_time(self, iteration_time: float):
        """Log training time for a single iteration."""
        self.iteration_times.append(iteration_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitoring session.
        
        Returns:
            Dictionary with summary statistics
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            'elapsed_time_seconds': round(elapsed_time, 2),
            'total_iterations': len(self.iteration_times),
        }
        
        if self.iteration_times:
            summary['avg_iteration_time_ms'] = round(
                sum(self.iteration_times) / len(self.iteration_times) * 1000, 2
            )
            summary['min_iteration_time_ms'] = round(min(self.iteration_times) * 1000, 2)
            summary['max_iteration_time_ms'] = round(max(self.iteration_times) * 1000, 2)
        
        if self.memory_history:
            latest_stats = self.memory_history[-1]
            summary['final_memory_stats'] = latest_stats.to_dict()
            
            # Find peak allocations across history
            all_peaks = [(s.peak_reserved_mb, s.peak_allocated_mb) for s in self.memory_history]
            if all_peaks:
                max_reserved = max(p[0] for p in all_peaks)
                max_allocated = max(p[1] for p in all_peaks)
                summary['absolute_peak_reserved_mb'] = round(max_reserved, 2)
                summary['absolute_peak_allocated_mb'] = round(max_allocated, 2)
        
        return summary
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("RESOURCE MONITORING SUMMARY")
        print("="*60)
        print(f"Elapsed Time: {summary['elapsed_time_seconds']} seconds")
        print(f"Total Iterations: {summary['total_iterations']}")
        
        if 'avg_iteration_time_ms' in summary:
            print(f"Avg Iteration Time: {summary['avg_iteration_time_ms']} ms")
            print(f"Min Iteration Time: {summary['min_iteration_time_ms']} ms")
            print(f"Max Iteration Time: {summary['max_iteration_time_ms']} ms")
        
        if 'final_memory_stats' in summary:
            stats = summary['final_memory_stats']
            print(f"\nFinal GPU Memory:")
            print(f"  Peak Reserved: {stats['peak_reserved_mb']} MB")
            print(f"  Peak Allocated: {stats['peak_allocated_mb']} MB")
            print(f"  Current Reserved: {stats['current_reserved_mb']} MB")
            print(f"  Current Allocated: {stats['current_allocated_mb']} MB")
        
        print("="*60 + "\n")
        
        return summary
    
    def save_summary(self, filepath: str):
        """
        Save monitoring summary to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        summary = self.get_summary()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Resource monitoring summary saved to: {filepath}")


def get_memory_stats(device: str = 'cuda') -> Optional[Dict[str, float]]:
    """
    Quick utility to get current memory stats as dictionary.
    
    Args:
        device: Device to monitor
        
    Returns:
        Dict with memory stats in MB or None if not CUDA
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    
    return {
        'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
        'current_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'current_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
    }


def print_memory_stats(label: str = "Memory Stats", device: str = 'cuda'):
    """
    Utility function to print memory stats with label.
    
    Args:
        label: Label for the memory stats printout
        device: Device to monitor
    """
    stats = get_memory_stats(device)
    if stats:
        print(f"\n{label}:")
        print(f"  Peak Reserved: {stats['peak_reserved_mb']:.2f} MB")
        print(f"  Peak Allocated: {stats['peak_allocated_mb']:.2f} MB")
        print(f"  Current Reserved: {stats['current_reserved_mb']:.2f} MB")
        print(f"  Current Allocated: {stats['current_allocated_mb']:.2f} MB\n")
    else:
        print(f"{label}: CUDA not available\n")


def reset_cuda_memory():
    """Reset CUDA memory cache and peak stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
