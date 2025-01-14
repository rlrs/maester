import wandb
import numpy as np
from scipy.optimize import curve_fit
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
    
    # Group runs by parameterization, width, num_steps, learning rate, and seed
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    skipped_runs = []
    
    for run in runs:
        print(f"\nProcessing run: {run.name}")
        config = run.config
        
        # Extract key parameters
        width = config.get('model_width')
        lr = config.get('opt_cfg', {}).get('lr')
        num_steps = config.get('train_num_steps')
        parameterization = config.get('parameterization', 'mup')
        
        if any(param is None for param in [width, lr, num_steps]):
            print(f"  WARNING: Skipping run - missing required parameters")
            skipped_runs.append(run.name)
            continue
            
        print(f"  Width: {width}, LR: {lr}, Steps: {num_steps}, Param: {parameterization}")
        
        # Get final smoothed training loss
        history = run.history(keys=['loss/global_avg'])
        if len(history) == 0:
            print(f"  WARNING: No loss data found")
            skipped_runs.append(run.name)
            continue
            
        smoothed_loss = history['loss/global_avg'].ewm(alpha=0.9).mean()
        final_loss = smoothed_loss.iloc[-1]
        
        data_dict[parameterization][width][num_steps][lr].append(final_loss)
    
    print("\n=== Data Loading Summary ===")
    print(f"Total runs processed: {len(runs)}")
    print(f"Runs skipped: {len(skipped_runs)}")
    
    return data_dict

def power_law_fit(x, a, b):
    """Power law function for fitting optimal learning rates"""
    return a * (x ** b)

def fit_optimal_lr_curve(steps, lrs):
    """Fit a power law to optimal learning rates"""
    try:
        popt, _ = curve_fit(power_law_fit, steps, lrs, p0=[1, -0.5])
        return popt
    except RuntimeError:
        print("Warning: Could not fit power law to optimal learning rates")
        return None

def plot_lr_sweep(data_dict, save_dir=None):
    """Plot learning rate sweep results with num_steps consideration"""
    sns.set(style='whitegrid')
    parameterizations = [('sp', 'SP'), ('mup', 'μP')]
    
    for param_name, param_label in parameterizations:
        if param_name not in data_dict:
            continue
            
        param_data = data_dict[param_name]
        widths = sorted(param_data.keys())
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Learning rate sweeps for different num_steps
        for width in widths:
            steps_data = param_data[width]
            color_helper = MplColorHelper('viridis', 0, len(steps_data)-1)
            
            optimal_lrs = []
            optimal_steps = []
            
            for step_idx, (num_steps, lr_data) in enumerate(sorted(steps_data.items())):
                lrs = sorted(lr_data.keys())
                losses = [np.mean(lr_data[lr]) for lr in lrs]
                
                ax1.plot(lrs, losses, label=f'{num_steps} steps', 
                        color=color_helper.get_rgb(step_idx), marker='o')
                
                # Store optimal learning rate
                best_idx = np.argmin(losses)
                optimal_lrs.append(lrs[best_idx])
                optimal_steps.append(num_steps)
        
            # Plot 2: Fit power law to optimal learning rates
            if optimal_lrs:
                ax2.scatter(optimal_steps, optimal_lrs, label=f'Width {width}')
                
                # Fit and plot power law
                popt = fit_optimal_lr_curve(optimal_steps, optimal_lrs)
                if popt is not None:
                    x_fit = np.logspace(np.log10(min(optimal_steps)), 
                                      np.log10(max(optimal_steps)), 100)
                    y_fit = power_law_fit(x_fit, *popt)
                    ax2.plot(x_fit, y_fit, '--', 
                            label=f'Fit: lr ∝ steps^{popt[1]:.2f}')
        
        # Format plots
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('Learning rate')
        ax1.set_ylabel('Train Loss')
        ax1.set_title(f'{param_label} Learning Rate Sweeps')
        ax1.legend(title='Training Steps')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Number of Training Steps')
        ax2.set_ylabel('Optimal Learning Rate')
        ax2.set_title(f'{param_label} Optimal Learning Rate Scaling')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_dir is not None:
            from pathlib import Path
            save_path = Path(save_dir) / f'lr_sweep_{param_name}.png'
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze learning rate sweep from wandb runs')
    parser.add_argument('--project', type=str, required=True, help='Wandb project name')
    parser.add_argument('--entity', type=str, required=True, help='Wandb entity/username')
    parser.add_argument('--save-dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    data = fetch_wandb_data(args.project, args.entity)
    if data:
        fig = plot_lr_sweep(data, save_dir=args.save_dir)
        plt.show()