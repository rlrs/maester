import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl
from collections import defaultdict

class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def fetch_wandb_data(project_name, entity_name=None):
    """
    Fetch activation data from wandb project runs.
    Returns a dictionary mapping widths to run data.
    """
    print("\n=== Fetching WandB Data ===")
    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}" if entity_name else project_name)
    print(f"Found {len(runs)} total runs")
    
    # Group runs by width
    width_data = defaultdict(list)
    skipped_runs = []
    
    # First, identify activation metrics
    sample_run = runs[0]
    history = sample_run.history()
    activation_metrics = [col for col in history.columns if col.startswith('act/')]
    print(f"\nFound activation metrics: {activation_metrics}")
    
    for run in runs:
        print(f"\nProcessing run: {run.name}")
        config = run.config
        width = config.get('model_width')
        
        if width is None:
            print(f"  WARNING: Skipping run - missing width in config")
            skipped_runs.append(run.name)
            continue
            
        print(f"  Width: {width}")
        history = run.history(keys=activation_metrics)
        
        if len(history) == 0:
            print(f"  WARNING: No data found in history")
            skipped_runs.append(run.name)
            continue
            
        print(f"  Found {len(history)} timesteps of data")
        width_data[width].append(history)
    
    print("\n=== Data Loading Summary ===")
    print(f"Total runs processed: {len(runs)}")
    print(f"Runs skipped: {len(skipped_runs)}")
    if skipped_runs:
        print("Skipped runs:")
        for run in skipped_runs:
            print(f"  - {run}")
    
    print("\nWidths found:")
    for width in sorted(width_data.keys()):
        print(f"  width={width}: {len(width_data[width])} runs")
        
    return width_data, activation_metrics

def plot_activation_scaling(width_data, metrics, t_max=10, save_dir=None):
    """
    Plot activation scaling data.
    """
    print("\n=== Creating Activation Scaling Plot ===")
    
    # Define layer types and their plotting parameters, matching original
    layer_info = [
        ('act/tok_embed_abs_mean', 'Word Embedding', None),
        ('act/attn_abs_mean', 'Attention', None),
        ('act/ffn_abs_mean', 'FFN', None),
        ('act/output_abs_mean', 'Output', None),
    ]
    
    # Verify we have all expected metrics
    available_metrics = set(metrics)
    expected_metrics = set(metric for metric, _, _ in layer_info)
    missing_metrics = expected_metrics - available_metrics
    if missing_metrics:
        print(f"WARNING: Missing expected metrics: {missing_metrics}")
    
    # Get sorted list of widths
    widths = sorted(width_data.keys())
    print(f"Plotting for widths: {widths}")
    
    # Setup plot
    sns.set_theme(style='whitegrid')
    color_helper = MplColorHelper('coolwarm', 0, t_max)
    n_cols = len(layer_info)
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 3))
    
    if n_cols == 1:
        axes = [axes]
    
    # For each layer type
    for layer_idx, (metric_name, layer_name, ylims) in enumerate(layer_info):
        if metric_name not in available_metrics:
            print(f"Skipping {layer_name} - metric not found")
            continue
            
        print(f"\nProcessing {layer_name}:")
        ax = axes[layer_idx]
        
        # For each timestep
        for t in range(t_max):
            means = []
            stderrs = []
            
            # For each width
            for width in widths:
                # Collect values for this width, metric, and timestep
                values = []
                for run_history in width_data[width]:
                    if len(run_history) > t:
                        if metric_name in run_history.columns:
                            val = run_history[metric_name].iloc[t]
                            if not np.isnan(val):
                                values.append(val)
                
                if values:
                    values = np.array(values)
                    means.append(np.mean(values))
                    stderrs.append(np.std(values, ddof=1) / np.sqrt(len(values)))
                    print(f"  Step {t+1}, Width {width}: "
                          f"mean={np.mean(values):.2e}, "
                          f"stderr={np.std(values, ddof=1) / np.sqrt(len(values)):.2e}, "
                          f"n={len(values)}")
            
            if means:  # Only plot if we have data
                means = np.array(means)
                stderrs = np.array(stderrs)
                ax.plot(widths, means, label=f'Step {t+1}', color=color_helper.get_rgb(t), marker='.')
                ax.fill_between(widths, means-stderrs, means+stderrs, color=color_helper.get_rgb(t), alpha=0.2)
        
        # Format plot
        ax.set_title(layer_name)
        ax.set_xlabel('Width')
        if layer_idx == 0:
            ax.set_ylabel('np.abs(activation).mean()')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        if ylims:
            ax.set_ylim(*ylims)
    
    plt.tight_layout()
    
    if save_dir is not None:
        from pathlib import Path
        import time
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = save_dir / f"activation_scaling_{timestamp}.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze activation scaling from wandb runs')
    parser.add_argument('--project', type=str, required=True, help='Wandb project name')
    parser.add_argument('--entity', type=str, required=True, help='Wandb entity/username')
    parser.add_argument('--save-dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--t-max', type=int, default=10, help='Maximum number of timesteps to plot')
    
    args = parser.parse_args()
    
    print(f"Starting analysis for project: {args.project}")
    width_data, metrics = fetch_wandb_data(args.project, args.entity)
    
    if not width_data:
        print("No data found! Exiting...")
        exit(1)
        
    print("\nCreating plots...")
    fig = plot_activation_scaling(width_data, metrics, t_max=args.t_max, save_dir=args.save_dir)
    
    plt.show()