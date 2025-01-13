import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
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
    Fetch learning rate sweep data from wandb project runs.
    Returns a dictionary mapping parameterizations to run data.
    """
    print("\n=== Fetching WandB Data ===")
    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}" if entity_name else project_name)
    print(f"Found {len(runs)} total runs")
    
    # Group runs by parameterization, width, learning rate, and seed
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped_runs = []
    
    for run in runs:
        print(f"\nProcessing run: {run.name}")
        config = run.config
        
        # Extract key parameters
        width = config.get('model_width')
        lr = config.get('opt_cfg').get('lr')
        parameterization = config.get('parameterization', 'mup')  # default to mup if not specified
        
        if width is None or lr is None:
            print(f"  WARNING: Skipping run - missing width or learning rate")
            skipped_runs.append(run.name)
            continue
            
        print(f"  Width: {width}, LR: {lr}, Param: {parameterization}")
        
        # Get final smoothed training loss
        history = run.history(keys=['loss/global_avg'])
        if len(history) == 0:
            print(f"  WARNING: No loss data found")
            skipped_runs.append(run.name)
            continue
            
        # Calculate EWM average of loss with alpha=0.9
        smoothed_loss = history['loss/global_avg'].ewm(alpha=0.9).mean()
        final_loss = smoothed_loss.iloc[-1]
        
        data_dict[parameterization][width][lr].append(final_loss)
    
    print("\n=== Data Loading Summary ===")
    print(f"Total runs processed: {len(runs)}")
    print(f"Runs skipped: {len(skipped_runs)}")
    
    print("\nParameterizations found:")
    for param in data_dict:
        print(f"  {param}:")
        for width in data_dict[param]:
            print(f"    Width {width}: {len(set(data_dict[param][width].keys()))} learning rates")
            
    return data_dict

def plot_lr_sweep(data_dict, save_dir=None):
    """
    Plot learning rate sweep results.
    """
    print("\n=== Creating Learning Rate Sweep Plot ===")
    
    parameterizations = [
        ('sp', r'SP'),
        ('mup', r'$\mu$P'),
    ]
    
    # Setup plot
    sns.set(style='whitegrid')
    n_cols = len([p for p in parameterizations if p[0] in data_dict])
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    if n_cols == 1:
        axes = [axes]
    
    ax_idx = 0
    for parameterization, parameterization_str in parameterizations:
        if parameterization not in data_dict:
            print(f"No data found for parameterization: {parameterization}")
            continue
            
        print(f"\nProcessing parameterization: {parameterization}")
        ax = axes[ax_idx]
        
        # Get sorted list of widths and create color mapping
        widths = sorted(data_dict[parameterization].keys())
        color_helper = MplColorHelper('viridis', 0, len(widths)-1)
        
        optimal_lrs = []
        optimal_losses = []
        
        # Plot for each width
        for width_idx, width in enumerate(widths):
            print(f"  Processing width {width}")
            width_data = data_dict[parameterization][width]
            
            # Prepare data for plotting
            lrs = sorted(width_data.keys())
            losses = [width_data[lr] for lr in lrs]
            avg_losses = [np.mean(l) for l in losses]
            sem_losses = [np.std(l, ddof=1) / np.sqrt(len(l)) if len(l) > 1 else 0 for l in losses]
            
            # Plot mean and standard error
            ax.plot(lrs, avg_losses, label=width, marker='o', 
                   color=color_helper.get_rgb(width_idx))
            ax.fill_between(lrs, 
                          np.array(avg_losses) - np.array(sem_losses),
                          np.array(avg_losses) + np.array(sem_losses),
                          color=color_helper.get_rgb(width_idx), alpha=0.33)
            
            # Find optimal learning rate
            if avg_losses:
                optimum_idx = np.argmin(avg_losses)
                optimal_lrs.append(lrs[optimum_idx])
                optimal_losses.append(avg_losses[optimum_idx])
        
        # Plot optimal points
        if optimal_lrs:
            ax.plot(optimal_lrs, optimal_losses, color='red', linestyle='none', marker='o')
        
        # Format plot
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Learning rate')
        ax.set_title(parameterization_str)
        ax.set_ylim(3.5, 5.5)
        
        if ax_idx == 0:
            ax.legend(title='Width')
            ax.set_ylabel('Train Loss')
        else:
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis='y', length=0, width=0)
        
        ax_idx += 1
    
    plt.tight_layout()
    
    if save_dir is not None:
        from pathlib import Path
        import time
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = save_dir / f"lr_sweep_{timestamp}.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze learning rate sweep from wandb runs')
    parser.add_argument('--project', type=str, required=True, help='Wandb project name')
    parser.add_argument('--entity', type=str, required=True, help='Wandb entity/username')
    parser.add_argument('--save-dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    print(f"Starting analysis for project: {args.project}")
    data = fetch_wandb_data(args.project, args.entity)
    
    if not data:
        print("No data found! Exiting...")
        exit(1)
        
    print("\nCreating plots...")
    fig = plot_lr_sweep(data, save_dir=args.save_dir)
    
    plt.show()