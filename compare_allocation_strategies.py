import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Directory containing results
RESULTS_DIR = "results"
STRATEGIES = ["even", "more_ll", "more_hf"]
IMAGEID = "kodim01"

def load_results(strategy):
    """Load results JSON for a given strategy"""
    results_file = os.path.join(RESULTS_DIR, f"dwt_adaptive_{strategy}", f"{IMAGEID}_{strategy}_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def plot_parameter_allocation():
    """Plot parameter allocation across bands for each strategy"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, strategy in enumerate(STRATEGIES):
        results = load_results(strategy)
        if not results:
            print(f"No results found for {strategy}")
            continue
        
        ax = axes[idx]
        bands = ['LL', 'LH', 'HL', 'HH']
        params = [results['bands'][band]['params'] for band in bands]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        bars = ax.bar(bands, params, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{param:,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Parameters', fontsize=12, fontweight='bold')
        ax.set_title(f'{strategy.replace("_", " ").title()}\nTotal: {results["overall"]["total_parameters"]:,} params', 
                    fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(params) * 1.15)
    
    plt.tight_layout()
    plt.savefig('result_imgs/allocation_comparison_params.png', dpi=300, bbox_inches='tight')
    print("Saved: result_imgs/allocation_comparison_params.png")
    plt.close()

def plot_psnr_comparison():
    """Plot PSNR comparison across strategies"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect data
    strategy_names = []
    overall_psnr = []
    band_psnr = {band: [] for band in ['LL', 'LH', 'HL', 'HH']}
    
    for strategy in STRATEGIES:
        results = load_results(strategy)
        if not results:
            continue
        
        strategy_names.append(strategy.replace("_", " ").title())
        overall_psnr.append(results['overall']['image_space_psnr'])
        
        for band in ['LL', 'LH', 'HL', 'HH']:
            band_psnr[band].append(results['bands'][band]['psnr'])
    
    # Plot 1: Overall Image PSNR
    bars = ax1.bar(strategy_names, overall_psnr, color=['#3498db', '#e74c3c', '#2ecc71'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, psnr in zip(bars, overall_psnr):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{psnr:.2f} dB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Image PSNR', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(overall_psnr) * 1.1)
    
    # Plot 2: Per-Band PSNR
    x = np.arange(len(strategy_names))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (band, color) in enumerate(zip(['LL', 'LH', 'HL', 'HH'], colors)):
        offset = (i - 1.5) * width
        ax2.bar(x + offset, band_psnr[band], width, label=band, 
               color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Band PSNR', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_names)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('result_imgs/allocation_comparison_psnr.png', dpi=300, bbox_inches='tight')
    print("Saved: result_imgs/allocation_comparison_psnr.png")
    plt.close()

def plot_efficiency_metrics():
    """Plot model size and compression ratio comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    strategy_names = []
    model_sizes = []
    compression_ratios = []
    psnr_values = []
    
    for strategy in STRATEGIES:
        results = load_results(strategy)
        if not results:
            continue
        
        strategy_names.append(strategy.replace("_", " ").title())
        model_sizes.append(results['overall']['total_model_size_kb'])
        compression_ratios.append(results['overall']['compression_ratio'])
        psnr_values.append(results['overall']['image_space_psnr'])
    
    # Plot 1: Model Size
    bars = ax1.bar(strategy_names, model_sizes, color=['#3498db', '#e74c3c', '#2ecc71'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, size in zip(bars, model_sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} kB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Model Size (kB)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Model Size', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(model_sizes) * 1.15)
    
    # Plot 2: PSNR vs Compression Ratio
    scatter = ax2.scatter(compression_ratios, psnr_values, 
                         c=['#3498db', '#e74c3c', '#2ecc71'], 
                         s=300, alpha=0.8, edgecolors='black', linewidth=2)
    
    for i, name in enumerate(strategy_names):
        # Add PSNR value annotation next to the point
        ax2.annotate(f'{psnr_values[i]:.2f} dB', 
                    (compression_ratios[i], psnr_values[i]),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        # Add strategy name
        ax2.annotate(name, (compression_ratios[i], psnr_values[i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Quality vs Compression', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('result_imgs/allocation_comparison_efficiency.png', dpi=300, bbox_inches='tight')
    print("Saved: result_imgs/allocation_comparison_efficiency.png")
    plt.close()

def plot_architecture_visualization():
    """Visualize architecture allocation as stacked representation"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = 0
    colors = {'LL': '#3498db', 'LH': '#e74c3c', 'HL': '#2ecc71', 'HH': '#f39c12'}
    
    for strategy in STRATEGIES:
        results = load_results(strategy)
        if not results:
            continue
        
        x_offset = 0
        total_params = results['overall']['total_parameters']
        
        for band in ['LL', 'LH', 'HL', 'HH']:
            params = results['bands'][band]['params']
            width = params / total_params * 10  # Normalize to width of 10
            
            rect = Rectangle((x_offset, y_pos), width, 0.8, 
                           facecolor=colors[band], edgecolor='black', 
                           linewidth=1.5, alpha=0.8)
            ax.add_patch(rect)
            
            # Add text if width is large enough
            if width > 1:
                ax.text(x_offset + width/2, y_pos + 0.4, 
                       f'{band}\n{params:,}',
                       ha='center', va='center', fontsize=9, fontweight='bold')
            
            x_offset += width
        
        # Add strategy label
        ax.text(-0.5, y_pos + 0.4, strategy.replace("_", " ").title(), 
               ha='right', va='center', fontsize=11, fontweight='bold')
        
        # Add total PSNR
        psnr = results['overall']['image_space_psnr']
        ax.text(10.5, y_pos + 0.4, f'{psnr:.2f} dB', 
               ha='left', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        y_pos += 1
    
    ax.set_xlim(-2, 12)
    ax.set_ylim(-0.2, y_pos + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Parameter Distribution Across Strategies', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [Rectangle((0,0),1,1, facecolor=colors[band], edgecolor='black', 
                                linewidth=1.5, alpha=0.8, label=band) 
                      for band in ['LL', 'LH', 'HL', 'HH']]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
             ncol=4, fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('result_imgs/allocation_comparison_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved: result_imgs/allocation_comparison_architecture.png")
    plt.close()

def generate_comparison_table():
    """Generate a summary table of all strategies"""
    print("\n" + "="*80)
    print("ALLOCATION STRATEGY COMPARISON")
    print("="*80)
    print(f"{'Strategy':<15} {'Total Params':<15} {'Model Size':<15} {'PSNR':<10} {'Compression':<12}")
    print("-"*80)
    
    for strategy in STRATEGIES:
        results = load_results(strategy)
        if not results:
            continue
        
        name = strategy.replace("_", " ").title()
        params = results['overall']['total_parameters']
        size = results['overall']['total_model_size_kb']
        psnr = results['overall']['image_space_psnr']
        compression = results['overall']['compression_ratio']
        
        print(f"{name:<15} {params:<15,} {size:<15.2f} {psnr:<10.2f} {compression:<12.2f}x")
    
    print("="*80)
    
    # Per-band breakdown
    print("\nPER-BAND PSNR (dB)")
    print("-"*80)
    print(f"{'Strategy':<15} {'LL':<10} {'LH':<10} {'HL':<10} {'HH':<10}")
    print("-"*80)
    
    for strategy in STRATEGIES:
        results = load_results(strategy)
        if not results:
            continue
        
        name = strategy.replace("_", " ").title()
        ll_psnr = results['bands']['LL']['psnr']
        lh_psnr = results['bands']['LH']['psnr']
        hl_psnr = results['bands']['HL']['psnr']
        hh_psnr = results['bands']['HH']['psnr']
        
        print(f"{name:<15} {ll_psnr:<10.2f} {lh_psnr:<10.2f} {hl_psnr:<10.2f} {hh_psnr:<10.2f}")
    
    print("="*80 + "\n")

def main():
    # Create output directory
    os.makedirs('result_imgs', exist_ok=True)
    
    # Generate all plots
    print("Generating comparison plots...")
    plot_parameter_allocation()
    plot_psnr_comparison()
    plot_efficiency_metrics()
    plot_architecture_visualization()
    
    # Generate comparison table
    generate_comparison_table()
    
    print("\nAll comparison plots generated successfully!")

if __name__ == "__main__":
    main()
