"""
Performance Audit Tools

Automated checks for:
1. Gradient flow analysis
2. Weight initialization audit
3. Data loading bottlenecks
4. Memory profiling
5. Computational efficiency
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import psutil
import warnings


class GradientFlowAuditor:
    """Analyze gradient flow through the network."""

    def __init__(self, model: nn.Module):
        """
        Initialize auditor.

        Args:
            model: PyTorch model to audit
        """
        self.model = model
        self.gradient_stats = {}

    def check_gradient_flow(
        self,
        loss: torch.Tensor,
        threshold: float = 1e-7
    ) -> Dict[str, Dict]:
        """
        Check for vanishing/exploding gradients.

        Args:
            loss: Loss tensor (after backward())
            threshold: Minimum gradient magnitude

        Returns:
            Dictionary with gradient statistics per layer
        """
        stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()
                grad_std = param.grad.std().item()

                stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'max': grad_max,
                    'std': grad_std,
                    'vanishing': grad_mean < threshold,
                    'exploding': grad_max > 10.0
                }

        self.gradient_stats = stats
        return stats

    def print_report(self):
        """Print gradient flow report."""
        print("\n" + "=" * 80)
        print("GRADIENT FLOW AUDIT")
        print("=" * 80)

        if not self.gradient_stats:
            print("⚠ No gradient statistics available. Run check_gradient_flow first.")
            return

        # Count issues
        n_vanishing = sum(1 for s in self.gradient_stats.values() if s['vanishing'])
        n_exploding = sum(1 for s in self.gradient_stats.values() if s['exploding'])

        print(f"\nTotal parameters: {len(self.gradient_stats)}")
        print(f"Vanishing gradients: {n_vanishing}")
        print(f"Exploding gradients: {n_exploding}")

        # Print detailed stats
        print("\n" + "-" * 80)
        print(f"{'Parameter':<40} {'Norm':>12} {'Mean':>12} {'Status':<15}")
        print("-" * 80)

        for name, stats in self.gradient_stats.items():
            status = "✓ OK"
            if stats['vanishing']:
                status = "⚠ VANISHING"
            elif stats['exploding']:
                status = "⚠ EXPLODING"

            print(f"{name:<40} {stats['norm']:>12.6f} {stats['mean']:>12.6f} {status:<15}")

        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        if n_vanishing > 0:
            print("⚠ Vanishing gradients detected. Consider:")
            print("  - Adding skip connections (Highway networks)")
            print("  - Using ReLU/GELU instead of Tanh/Sigmoid")
            print("  - Reducing network depth")
            print("  - Using gradient clipping (already enabled)")

        if n_exploding > 0:
            print("⚠ Exploding gradients detected. Consider:")
            print("  - Gradient clipping (check current threshold)")
            print("  - Reducing learning rate")
            print("  - Weight initialization (see initialization audit)")

        if n_vanishing == 0 and n_exploding == 0:
            print("✓ Gradient flow looks healthy!")


class InitializationAuditor:
    """Audit weight initialization for saturation issues."""

    def __init__(self, model: nn.Module):
        """
        Initialize auditor.

        Args:
            model: PyTorch model to audit
        """
        self.model = model
        self.init_stats = {}

    def check_initialization(self) -> Dict[str, Dict]:
        """
        Check weight initialization.

        Returns:
            Dictionary with initialization statistics
        """
        stats = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                data = param.data
                stats[name] = {
                    'mean': data.mean().item(),
                    'std': data.std().item(),
                    'min': data.min().item(),
                    'max': data.max().item(),
                    'shape': tuple(data.shape)
                }

                # Check for saturation (for gates with sigmoid/tanh)
                if 'gate' in name.lower() or 'bias' in name.lower():
                    # For sigmoid gates, check if biases push toward saturation
                    if 'bias' in name.lower():
                        if abs(data.mean().item()) > 2.0:
                            stats[name]['warning'] = 'Bias may saturate sigmoid'

        self.init_stats = stats
        return stats

    def print_report(self):
        """Print initialization report."""
        print("\n" + "=" * 80)
        print("INITIALIZATION AUDIT")
        print("=" * 80)

        if not self.init_stats:
            print("⚠ No initialization statistics. Run check_initialization first.")
            return

        print("\n" + "-" * 80)
        print(f"{'Parameter':<40} {'Mean':>12} {'Std':>12} {'Shape':<20}")
        print("-" * 80)

        for name, stats in self.init_stats.items():
            shape_str = str(stats['shape'])
            print(f"{name:<40} {stats['mean']:>12.6f} {stats['std']:>12.6f} {shape_str:<20}")

            if 'warning' in stats:
                print(f"  ⚠ {stats['warning']}")

        # General recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Check for zero initialization
        zero_inits = [name for name, s in self.init_stats.items()
                     if abs(s['std']) < 1e-6 and 'bias' not in name.lower()]

        if zero_inits:
            print(f"⚠ {len(zero_inits)} parameters have near-zero std (may not learn):")
            for name in zero_inits[:5]:  # Show first 5
                print(f"  - {name}")

        # Check for very large initialization
        large_inits = [name for name, s in self.init_stats.items()
                      if s['std'] > 1.0]

        if large_inits:
            print(f"⚠ {len(large_inits)} parameters have std > 1.0 (may be too large):")
            for name in large_inits[:5]:
                print(f"  - {name}")

        print("\nFor gating mechanisms (Tanh/Sigmoid):")
        print("  - Use Xavier/Glorot initialization")
        print("  - Initialize biases to small positive values (e.g., 1.0) for gates")
        print("  - This prevents saturation at initialization")


class DataLoaderAuditor:
    """Audit data loading efficiency."""

    def __init__(self):
        """Initialize auditor."""
        self.timing_stats = []

    def benchmark_dataloader(
        self,
        dataloader,
        n_batches: int = 100
    ) -> Dict:
        """
        Benchmark dataloader speed.

        Args:
            dataloader: PyTorch DataLoader
            n_batches: Number of batches to test

        Returns:
            Dictionary with timing statistics
        """
        print("\n" + "=" * 80)
        print("DATALOADER BENCHMARK")
        print("=" * 80)

        times = []
        cpu_usage = []
        mem_usage = []

        process = psutil.Process()

        print(f"\nTesting {n_batches} batches...")

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            start = time.time()

            # Simulate minimal processing
            _ = batch

            elapsed = time.time() - start
            times.append(elapsed)

            # Resource usage
            cpu_usage.append(process.cpu_percent())
            mem_info = process.memory_info()
            mem_usage.append(mem_info.rss / 1024 / 1024)  # MB

        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'mean_cpu': np.mean(cpu_usage),
            'mean_memory_mb': np.mean(mem_usage)
        }

        # Print report
        print(f"\nBatch loading time:")
        print(f"  Mean: {stats['mean_time']*1000:.2f} ms")
        print(f"  Std: {stats['std_time']*1000:.2f} ms")
        print(f"  Median: {stats['median_time']*1000:.2f} ms")
        print(f"  Min: {stats['min_time']*1000:.2f} ms")
        print(f"  Max: {stats['max_time']*1000:.2f} ms")

        print(f"\nResource usage:")
        print(f"  CPU: {stats['mean_cpu']:.1f}%")
        print(f"  Memory: {stats['mean_memory_mb']:.1f} MB")

        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        if stats['mean_time'] > 0.1:
            print("⚠ Data loading is slow (>100ms per batch). Consider:")
            print("  - Increase num_workers in DataLoader")
            print("  - Use pin_memory=True for GPU training")
            print("  - Enable persistent_workers=True")
            print("  - Check disk I/O (use SSD if possible)")

        if stats['mean_cpu'] > 80:
            print("⚠ High CPU usage. Consider:")
            print("  - Reduce num_workers")
            print("  - Simplify data preprocessing")

        if stats['mean_time'] < 0.01:
            print("✓ Data loading is fast! GPU may be the bottleneck.")

        return stats


class MemoryProfiler:
    """Profile memory usage during training."""

    def __init__(self, device: torch.device):
        """
        Initialize profiler.

        Args:
            device: Device to profile
        """
        self.device = device
        self.is_cuda = device.type == 'cuda'

    def get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        stats = {}

        if self.is_cuda:
            stats['allocated_mb'] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            stats['reserved_mb'] = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            stats['max_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            stats['rss_mb'] = mem_info.rss / 1024 / 1024

        return stats

    def print_report(self):
        """Print memory usage report."""
        print("\n" + "=" * 80)
        print("MEMORY USAGE")
        print("=" * 80)

        stats = self.get_memory_stats()

        if self.is_cuda:
            print(f"\nGPU Memory:")
            print(f"  Allocated: {stats['allocated_mb']:.1f} MB")
            print(f"  Reserved: {stats['reserved_mb']:.1f} MB")
            print(f"  Max Allocated: {stats['max_allocated_mb']:.1f} MB")

            # Recommendations
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024
            usage_pct = (stats['allocated_mb'] / total_memory) * 100

            print(f"\nUsage: {usage_pct:.1f}% of {total_memory:.0f} MB")

            if usage_pct > 90:
                print("\n⚠ High memory usage! Consider:")
                print("  - Reduce batch size")
                print("  - Enable gradient checkpointing")
                print("  - Use mixed precision training (FP16)")
        else:
            print(f"\nCPU Memory:")
            print(f"  RSS: {stats['rss_mb']:.1f} MB")


def run_full_audit(
    model: nn.Module,
    dataloader,
    device: torch.device,
    sample_batch: Optional[torch.Tensor] = None
):
    """
    Run complete performance audit.

    Args:
        model: Model to audit
        dataloader: DataLoader to test
        device: Device
        sample_batch: Optional sample batch for gradient flow test
    """
    print("\n" + "=" * 80)
    print("COMPLETE PERFORMANCE AUDIT")
    print("=" * 80)

    # 1. Initialization audit
    init_auditor = InitializationAuditor(model)
    init_auditor.check_initialization()
    init_auditor.print_report()

    # 2. Gradient flow audit (if sample provided)
    if sample_batch is not None:
        print("\n" + "=" * 80)
        print("Running gradient flow test...")
        grad_auditor = GradientFlowAuditor(model)

        model.train()
        x, y = sample_batch
        x, y = x.to(device), y.to(device)

        # Forward + backward
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]

        loss = nn.MSELoss()(output, y)
        loss.backward()

        grad_auditor.check_gradient_flow(loss)
        grad_auditor.print_report()

    # 3. Dataloader audit
    dataloader_auditor = DataLoaderAuditor()
    dataloader_auditor.benchmark_dataloader(dataloader, n_batches=50)

    # 4. Memory profiling
    mem_profiler = MemoryProfiler(device)
    mem_profiler.print_report()

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    # Test audit tools
    print("Testing Performance Audit Tools")
    print("=" * 80)

    from models.z2_seam_gated import Z2SeamGatedRNN
    from torch.utils.data import DataLoader, TensorDataset

    # Create model
    model = Z2SeamGatedRNN(input_dim=4, hidden_dim=64)

    # Create dummy data
    X = torch.randn(1000, 10, 4)
    y = torch.randn(1000, 4)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

    # Get sample batch
    sample_x, sample_y = next(iter(dataloader))

    # Run full audit
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    run_full_audit(
        model,
        dataloader,
        device,
        sample_batch=(sample_x, sample_y)
    )

    print("\n✓ Audit tools test completed!")
