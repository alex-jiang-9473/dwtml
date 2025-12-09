import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ---------------------------
# CONFIG
RESULTS_DIR = "results/dwt"
OUTPUT_DIR = "results/plots"
IMAGEID = "kodim01"  # Filter by image ID, or set to None for all images

# ---------------------------

def load_results(results_dir, image_id=None):
    """Load all JSON result files from the results directory."""
    pattern = os.path.join(results_dir, f"{image_id}_results_*.json" if image_id else "*_results_*.json")
    result_files = glob.glob(pattern)
    
    results = []
    for file in result_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"Loaded {len(results)} result files")
    return results

def plot_psnr_vs_model_size(results, output_dir):
    """Plot PSNR vs Model Size."""
    model_sizes = [r['model']['model_size_kb'] for r in results]
    psnr_image = [r['quality']['image_space_psnr'] for r in results]
    labels = [r['model']['model_architecture'] for r in results]
    
    plt.figure(figsize=(12, 7))
    plt.scatter(model_sizes, psnr_image, alpha=0.7, s=50)
    
    # Annotate each point with architecture
    for i, (x, y, label) in enumerate(zip(model_sizes, psnr_image, labels)):
        plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('Model Size (kB)')
    plt.ylabel('Image-space PSNR (dB)')
    plt.title('Image-space PSNR vs Model Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_vs_model_size.png'), dpi=150)
    plt.close()
    print("Saved: psnr_vs_model_size.png")

def plot_psnr_vs_iterations(results, output_dir):
    """Plot PSNR vs Model Architecture."""
    # Group by iteration count
    by_iters = defaultdict(lambda: {'archs': [], 'psnr': [], 'model_sizes': []})
    for r in results:
        iters = r['config']['iterations']
        by_iters[iters]['archs'].append(r['model']['model_architecture'])
        by_iters[iters]['psnr'].append(r['quality']['image_space_psnr'])
        by_iters[iters]['model_sizes'].append(r['model']['model_size_kb'])
    
    plt.figure(figsize=(14, 6))
    
    # Get all unique architectures and sort by model size
    all_archs = []
    arch_to_size = {}
    for r in results:
        arch = r['model']['model_architecture']
        if arch not in arch_to_size:
            arch_to_size[arch] = r['model']['model_size_kb']
            all_archs.append(arch)
    
    # Sort architectures by model size
    all_archs_sorted = sorted(all_archs, key=lambda a: arch_to_size[a])
    
    for iters, data in sorted(by_iters.items()):
        # Sort by architecture order (which is sorted by model size)
        arch_order = {arch: i for i, arch in enumerate(all_archs_sorted)}
        sorted_indices = sorted(range(len(data['archs'])), key=lambda i: arch_order.get(data['archs'][i], 999))
        sorted_archs = [data['archs'][i] for i in sorted_indices]
        sorted_psnr = [data['psnr'][i] for i in sorted_indices]
        
        x_positions = [arch_order[arch] for arch in sorted_archs]
        plt.plot(x_positions, sorted_psnr, marker='o', label=f'{iters} iters', alpha=0.7, linewidth=2)
        
        # Add PSNR value annotations on the points
        for x, y in zip(x_positions, sorted_psnr):
            plt.annotate(f'{y:.2f}', (x, y), xytext=(0, 5), textcoords='offset points',
                        fontsize=7, ha='center', alpha=0.6)
    
    plt.xticks(range(len(all_archs_sorted)), all_archs_sorted, rotation=45, ha='right')
    plt.xlabel('Model Architecture (Layers × Neurons)')
    plt.ylabel('Image-space PSNR (dB)')
    plt.title('PSNR vs Model Setup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_vs_model_setup.png'), dpi=150)
    plt.close()
    print("Saved: psnr_vs_model_setup.png")

def plot_compression_ratio_vs_psnr(results, output_dir):
    """Plot Compression Ratio vs PSNR (Rate-Distortion curve)."""
    compression_ratios = [r['compression']['compression_ratio'] for r in results]
    psnr_image = [r['quality']['image_space_psnr'] for r in results]
    bpp_fp16 = [r['compression']['fp16_bpp'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compression ratio vs PSNR
    ax1.scatter(compression_ratios, psnr_image, alpha=0.7, s=50)
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('Image-space PSNR (dB)')
    ax1.set_title('Rate-Distortion: Compression Ratio vs PSNR')
    ax1.grid(True, alpha=0.3)
    
    # BPP vs PSNR
    ax2.scatter(bpp_fp16, psnr_image, alpha=0.7, s=50, color='orange')
    ax2.set_xlabel('Bits Per Pixel (FP16)')
    ax2.set_ylabel('Image-space PSNR (dB)')
    ax2.set_title('Rate-Distortion: BPP vs PSNR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rate_distortion.png'), dpi=150)
    plt.close()
    print("Saved: rate_distortion.png")

def plot_training_time_vs_model_size(results, output_dir):
    """Plot Training Time vs Model Complexity."""
    model_params = [r['model']['total_parameters'] for r in results]
    training_time = [r['training']['training_time_sec'] for r in results]
    total_time = [r['training']['total_time_sec'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training time vs parameters
    ax1.scatter(model_params, training_time, alpha=0.7, s=50)
    ax1.set_xlabel('Total Parameters')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time vs Model Parameters')
    ax1.grid(True, alpha=0.3)
    
    # Total time vs parameters
    ax2.scatter(model_params, total_time, alpha=0.7, s=50, color='green')
    ax2.set_xlabel('Total Parameters')
    ax2.set_ylabel('Total Time (seconds)')
    ax2.set_title('Total Time vs Model Parameters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_model.png'), dpi=150)
    plt.close()
    print("Saved: time_vs_model.png")

def plot_coefficient_analysis(results, output_dir):
    """Plot coefficient statistics analysis."""
    coeff_counts = [r['coefficients']['total_count'] for r in results]
    coeff_stds = [r['coefficients']['std'] for r in results]
    psnr_image = [r['quality']['image_space_psnr'] for r in results]
    model_sizes = [r['model']['model_size_kb'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Coefficient std vs PSNR
    scatter1 = ax1.scatter(coeff_stds, psnr_image, c=model_sizes, 
                           cmap='viridis', alpha=0.7, s=50)
    ax1.set_xlabel('Coefficient Std Dev')
    ax1.set_ylabel('Image-space PSNR (dB)')
    ax1.set_title('Coefficient Variability vs Quality')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Model Size (kB)')
    
    # Model size vs coefficient count
    ax2.scatter(coeff_counts, model_sizes, alpha=0.7, s=50, color='purple')
    ax2.set_xlabel('Coefficient Count')
    ax2.set_ylabel('Model Size (kB)')
    ax2.set_title('Model Size vs Coefficient Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coefficient_analysis.png'), dpi=150)
    plt.close()
    print("Saved: coefficient_analysis.png")

def plot_architecture_comparison(results, output_dir):
    """Compare different architectures (layers x size)."""
    # Group by architecture
    by_arch = defaultdict(lambda: {
        'psnr': [], 'model_size': [], 'training_time': [], 
        'compression_ratio': [], 'bpp': []
    })
    
    for r in results:
        arch = r['model']['model_architecture']
        by_arch[arch]['psnr'].append(r['quality']['image_space_psnr'])
        by_arch[arch]['model_size'].append(r['model']['model_size_kb'])
        by_arch[arch]['training_time'].append(r['training']['training_time_sec'])
        by_arch[arch]['compression_ratio'].append(r['compression']['compression_ratio'])
        by_arch[arch]['bpp'].append(r['compression']['fp16_bpp'])
    
    # Calculate averages
    arch_stats = {}
    for arch, data in by_arch.items():
        arch_stats[arch] = {
            'psnr': np.mean(data['psnr']),
            'model_size': np.mean(data['model_size']),
            'training_time': np.mean(data['training_time']),
            'compression_ratio': np.mean(data['compression_ratio']),
            'bpp': np.mean(data['bpp'])
        }
    
    # Plot comparison
    archs = list(arch_stats.keys())
    psnrs = [arch_stats[a]['psnr'] for a in archs]
    model_sizes = [arch_stats[a]['model_size'] for a in archs]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # PSNR comparison
    ax1.bar(range(len(archs)), psnrs, alpha=0.7)
    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels(archs, rotation=45, ha='right')
    ax1.set_ylabel('Image-space PSNR (dB)')
    ax1.set_title('PSNR by Architecture')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Model size comparison
    ax2.bar(range(len(archs)), model_sizes, alpha=0.7, color='orange')
    ax2.set_xticks(range(len(archs)))
    ax2.set_xticklabels(archs, rotation=45, ha='right')
    ax2.set_ylabel('Model Size (kB)')
    ax2.set_title('Model Size by Architecture')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Compression ratio comparison
    comp_ratios = [arch_stats[a]['compression_ratio'] for a in archs]
    ax3.bar(range(len(archs)), comp_ratios, alpha=0.7, color='green')
    ax3.set_xticks(range(len(archs)))
    ax3.set_xticklabels(archs, rotation=45, ha='right')
    ax3.set_ylabel('Compression Ratio')
    ax3.set_title('Compression Ratio by Architecture')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Training time comparison
    train_times = [arch_stats[a]['training_time'] for a in archs]
    ax4.bar(range(len(archs)), train_times, alpha=0.7, color='red')
    ax4.set_xticks(range(len(archs)))
    ax4.set_xticklabels(archs, rotation=45, ha='right')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title('Training Time by Architecture')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_comparison.png'), dpi=150)
    plt.close()
    print("Saved: architecture_comparison.png")

def plot_gpu_memory_usage(results, output_dir):
    """Plot GPU memory usage analysis."""
    model_sizes = [r['model']['model_size_kb'] for r in results]
    model_params = [r['model']['total_parameters'] for r in results]
    gpu_memory = [r['hardware']['gpu_memory_mb'] for r in results]
    labels = [r['model']['model_architecture'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # GPU memory vs Model Size
    ax1.scatter(model_sizes, gpu_memory, alpha=0.7, s=50, color='red')
    for i, (x, y, label) in enumerate(zip(model_sizes, gpu_memory, labels)):
        ax1.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    ax1.set_xlabel('Model Size (kB)')
    ax1.set_ylabel('Peak GPU Memory (MB)')
    ax1.set_title('GPU Memory Usage vs Model Size')
    ax1.grid(True, alpha=0.3)
    
    # GPU memory vs Parameters
    ax2.scatter(model_params, gpu_memory, alpha=0.7, s=50, color='orange')
    for i, (x, y, label) in enumerate(zip(model_params, gpu_memory, labels)):
        ax2.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    ax2.set_xlabel('Total Parameters')
    ax2.set_ylabel('Peak GPU Memory (MB)')
    ax2.set_title('GPU Memory Usage vs Model Parameters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_memory_usage.png'), dpi=150)
    plt.close()
    print("Saved: gpu_memory_usage.png")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load results
    results = load_results(RESULTS_DIR, IMAGEID)
    
    if len(results) == 0:
        print("No results found! Run dwt_siren.py first to generate results.")
        return
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_psnr_vs_model_size(results, OUTPUT_DIR)
    plot_psnr_vs_iterations(results, OUTPUT_DIR)
    plot_compression_ratio_vs_psnr(results, OUTPUT_DIR)
    plot_training_time_vs_model_size(results, OUTPUT_DIR)
    plot_coefficient_analysis(results, OUTPUT_DIR)
    plot_architecture_comparison(results, OUTPUT_DIR)
    
    # GPU memory plot (only if GPU data available)
    if all('gpu_memory_mb' in r.get('hardware', {}) for r in results):
        if any(r['hardware']['gpu_memory_mb'] > 0 for r in results):
            plot_gpu_memory_usage(results, OUTPUT_DIR)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
