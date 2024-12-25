"""
CLI for managing a parameter sweep as a set of SLURM jobs
"""
from pathlib import Path
import itertools
import sys
import json
import subprocess
from typing import Dict, List, Any, Union, Optional

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass
import wandb
import seaborn


from maester.config import Config
from submit import validate_config, setup_job_dir, submit_job

@dataclass
class JobInfo:
    """Information about a sweep job"""
    name: str
    job_id: Optional[str]
    status: str
    wandb_run: Optional[str] = None
    last_metric: Optional[float] = None
    error: Optional[str] = None

def get_slurm_jobs(user: Optional[str] = None) -> Dict[str, str]:
    """Get all running/pending SLURM jobs for user"""
    cmd = ["squeue", "--format=%j|%i|%T", "--noheader"]
    if user:
        cmd.extend(["-u", user])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    jobs = {}
    for line in result.stdout.splitlines():
        name, job_id, status = line.strip().split("|")
        jobs[job_id] = {"name": name, "status": status}
    return jobs

def get_sweep_status(sweep_dir: Union[str, Path]) -> Dict[str, JobInfo]:
    """Get status of all jobs in a sweep"""
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        raise ValueError(f"Sweep directory {sweep_dir} does not exist")
        
    # Load submission summary
    summary_file = sweep_dir / "submission_summary.txt"
    job_map = {}  # name -> job_id
    if summary_file.exists():
        with open(summary_file) as f:
            for line in f:
                if not line or "Submitted jobs" in line:  # Skip header/empty lines
                    continue
                if ":" in line:
                    name, job_id = line.strip().split(": ")
                    job_map[name] = job_id
    
    # Get current SLURM status
    slurm_jobs = get_slurm_jobs()

    # Load sweep config to get wandb info
    sweep_config_file = sweep_dir / "sweep_config.json"
    wandb_entity = None
    project_name = None
    if sweep_config_file.exists():
        with open(sweep_config_file) as f:
            sweep_config = json.load(f)
            try:
                # Get base config wandb settings
                base_config = sweep_config.get('base_config', {})
                wandb_entity = base_config.get('wandb_entity')
                project_name = base_config.get('wandb_project')
            except Exception as e:
                print(f"Warning: Could not parse wandb info from sweep config: {e}")
    
    
    # Get wandb runs if possible
    wandb_runs = {}
    if wandb_entity and project_name:
        try:
            api = wandb.Api()
            project = api.project(entity=wandb_entity, name=project_name)
            # Get all runs in project
            runs = project.runs
            
            # Match runs to jobs based on config
            for run in runs:
                try:
                    # Get the job's parameter values from config
                    config = run.config
                    if not config:
                        continue
                        
                    # Build job name from parameters that match our naming scheme
                    params = {}
                    if 'flavor' in config:
                        params['flavor'] = config['flavor']
                    if 'opt_cfg' in config:
                        if 'lr' in config['opt_cfg']:
                            params['lr'] = config['opt_cfg']['lr']
                        if 'weight_decay' in config['opt_cfg']:
                            params['wd'] = config['opt_cfg']['weight_decay']
                    # Add other parameters as needed
                    
                    # Generate job name using same format as sweep
                    job_name = get_job_name(params)  # Using our naming function
                    wandb_runs[job_name] = {
                        'run_id': run.id,
                        'name': run.name,
                        'status': run.state,
                        'metrics': run.summary._json_dict if run.summary else {}
                    }
                except Exception as e:
                    print(f"Warning: Could not parse run {run.name}: {e}")
                    
        except Exception as e:
            print(f"Warning: Could not fetch wandb runs: {e}")
    
    # Build status for each job
    jobs = {}
    for job_dir in sweep_dir.glob("*"):
        if not job_dir.is_dir() or job_dir.name in {"logs", "checkpoints"}:
            continue
            
        name = job_dir.name
        job_id = job_map.get(name)
        
        # Check latest logs for errors
        error = None
        if job_id:
            log_file = job_dir / "logs" / f"{job_id}.err"
            if log_file.exists():
                with open(log_file) as f:
                    errors = [l for l in f if "Error" in l or "ERROR" in l]
                    if errors:
                        error = errors[-1].strip()
        
        # Determine status
        if job_id in slurm_jobs:
            status = slurm_jobs[job_id]["status"]
        else:
            # Check if completed successfully
            if (job_dir / "checkpoints" / "finished.txt").exists():
                status = "COMPLETED"
            elif error:
                status = "FAILED"
            else:
                status = "UNKNOWN"
        
        # Get wandb info
        wandb_run = None
        last_metric = None
        if name in runs:
            run = runs[name]
            wandb_run = run.id
            # Assuming you track some key metric
            try:
                last_metric = run.summary.get("loss/global_avg")
            except:
                pass
        
        jobs[name] = JobInfo(
            name=name,
            job_id=job_id,
            status=status,
            wandb_run=wandb_run,
            last_metric=last_metric,
            error=error
        )
    
    return jobs

def summarize_sweep(sweep_dir: Union[str, Path], output: Optional[str] = None) -> pd.DataFrame:
    """Generate summary of sweep results"""
    jobs = get_sweep_status(sweep_dir)
    
    # Convert to DataFrame for easier analysis
    records = []
    for job in jobs.values():
        # Parse parameters from job name
        params = {}
        for param in job.name.split("-"):
            if "lr" in param:
                params["learning_rate"] = float(param.replace("lr", ""))
            elif "wd" in param:
                params["weight_decay"] = float(param.replace("wd", ""))
            elif param in {"7b", "13b", "70b"}:
                params["model_size"] = param
            else:
                # Handle other parameters
                if "=" in param:
                    k, v = param.split("=")
                    params[k] = v
        
        record = {
            "job_name": job.name,
            "status": job.status,
            "metric": job.last_metric,
            **params
        }
        records.append(record)
    
    df = pd.DataFrame.from_records(records)
    
    if output:
        if output.endswith(".csv"):
            df.to_csv(output, index=False)
        else:
            df.to_pickle(output)
    
    return df

def cancel_sweep_jobs(sweep_dir: Union[str, Path]):
    """Cancel all running jobs in a sweep"""
    jobs = get_sweep_status(sweep_dir)
    running = [j.job_id for j in jobs.values() 
              if j.job_id and j.status in {"PENDING", "RUNNING"}]
    
    if running:
        print(f"Canceling {len(running)} jobs...")
        subprocess.run(["scancel"] + running)
    else:
        print("No running jobs found")

def retry_failed_jobs(sweep_dir: Union[str, Path], dry_run: bool = False):
    """Resubmit failed jobs in a sweep"""
    jobs = get_sweep_status(sweep_dir)
    failed = [name for name, job in jobs.items() 
             if job.status == "FAILED" or 
             (job.status == "UNKNOWN" and job.error)]
    
    if not failed:
        print("No failed jobs found")
        return
    
    print(f"Found {len(failed)} failed jobs")
    
    sweep_dir = Path(sweep_dir)
    for name in failed:
        job_dir = sweep_dir / name
        if dry_run:
            print(f"Would resubmit {name}")
        else:
            try:
                result = subprocess.run(
                    ["sbatch", str(job_dir / "slurm.sh")],
                    capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip().split()[-1]
                print(f"Resubmitted {name}: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to resubmit {name}: {e.stderr}")

def plot_sweep_results(sweep_dir: Union[str, Path], 
                      x: str, y: str,
                      hue: Optional[str] = None,
                      output: Optional[str] = None):
    """Plot sweep results"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df = summarize_sweep(sweep_dir)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.xscale('log' if 'lr' in x else 'linear')
    plt.yscale('log' if 'loss' in y else 'linear')
    
    if output:
        plt.savefig(output)
    else:
        plt.show()

class SweepValues(BaseModel):
    """Represents different ways to specify parameter values"""
    values: List[Any]
    
    @classmethod
    def from_list(cls, values: List[Any]) -> "SweepValues":
        return cls(values=values)
    
    @classmethod
    def linspace(cls, start: float, end: float, num: int) -> "SweepValues":
        values = [start + (end - start) * i / (num - 1) for i in range(num)]
        return cls(values=values)
    
    @classmethod
    def logspace(cls, start: float, end: float, num: int, base: int = 10) -> "SweepValues":
        import numpy as np
        values = np.logspace(start, end, num, base=base).tolist()
        return cls(values=values)

class SweepConfig(BaseSettings):
    """Configuration for parameter sweeps"""
    model_config = SettingsConfigDict(frozen=True, env_prefix="SWEEP_", cli_parse_args=False)
    
    # Base config to modify
    base_config: Config = Config()
    sweep_name: str
    
    # Parameter spaces defined as nested dicts matching config structure
    # e.g. {"flavor": ["7b", "13b"], "opt_cfg.lr": [1e-4, 3e-4, 1e-3]}
    parameter_space: Dict[str, Union[List[Any], SweepValues]]

def format_param_value(value: Any) -> str:
    """Format parameter values for job names"""
    if isinstance(value, float):
        return f"{value:.1e}".replace('-', 'n').replace('+', 'p')
    return str(value).replace('-', 'n')

def get_job_name(params: Dict[str, Any]) -> str:
    """Create readable job name from parameters"""
    parts = []
    for key, value in sorted(params.items()):  # Sort for consistent ordering
        short_key = key.split('.')[-1]
        
        # Special cases for common parameters
        if short_key == "lr":
            parts.append(f"lr{format_param_value(value)}")
        elif short_key == "weight_decay":
            parts.append(f"wd{format_param_value(value)}")
        elif short_key == "flavor":
            parts.append(str(value))
        else:
            parts.append(f"{short_key}{format_param_value(value)}")
            
    return "-".join(parts)

def set_nested_key(d: Dict[str, Any], key: str, value: Any):
    """Set a value in a nested dictionary using dot notation"""
    keys = key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

def generate_configs(sweep_cfg: SweepConfig) -> Dict[str, Config]:
    """Generate all valid configurations for the sweep"""
    # Convert any lists to SweepValues
    param_space = {
        k: SweepValues.from_list(v) if isinstance(v, list) else v 
        for k, v in sweep_cfg.parameter_space.items()
    }
    
    # Get all parameter combinations
    param_names = list(param_space.keys())
    param_values = [param_space[name].values for name in param_names]
    configs = {}
    
    
    for value_combo in itertools.product(*param_values):
        # Start with base config
        config_dict = sweep_cfg.base_config.model_dump()
        
        # Set job folder to sweep subdirectory
        config_dict["dump_dir"] = str(Path("jobs") / sweep_cfg.sweep_name)

        param_dict = {}
        for name, value in zip(param_names, value_combo):
            set_nested_key(config_dict, name, value)
            param_dict[name] = value
            
        config_dict["job_name"] = get_job_name(param_dict)
        
        try:
            cfg = Config(**config_dict)
            errors = validate_config(cfg)
            if not any(error for error in errors if "Warning" not in error):
                configs[config_dict["job_name"]] = cfg
        except Exception as e:
            print(f"Invalid config {config_dict['job_name']}: {e}")
            
    return configs

def run_sweep(sweep_cfg: SweepConfig, dry_run: bool = False):
    """Generate and submit sweep jobs"""
    configs = generate_configs(sweep_cfg)
    print(f"Generated {len(configs)} valid configurations")
    
    # Create sweep directory
    sweep_dir = Path("jobs") / sweep_cfg.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sweep config for reference
    with open(sweep_dir / "sweep_config.json", "w") as f:
        f.write(sweep_cfg.model_dump_json(indent=2))
    
    # Submit jobs
    submitted = []
    failed = []
    for job_name, cfg in configs.items():
        print(f"\nSubmitting {job_name}...")
        try:
            job_dir = setup_job_dir(cfg)
            if not dry_run:
                job_id = submit_job(job_dir)
                submitted.append((job_name, job_id))
                print(f"Submitted {job_name} ({job_id})")
            else:
                print(f"Would submit {job_name}")
        except Exception as e:
            print(f"Failed to submit {job_name}: {e}")
            failed.append(job_name)
    
    # Write submission summary
    with open(sweep_dir / "submission_summary.txt", "w") as f:
        f.write(f"Submitted jobs ({len(submitted)}):\n")
        for name, job_id in submitted:
            f.write(f"{name}: {job_id}\n")
        if failed:
            f.write(f"\nFailed jobs ({len(failed)}):\n")
            for name in failed:
                f.write(f"{name}\n")

console = Console()

def load_sweep_config(config_path: str) -> SweepConfig:
    """Load sweep configuration from a Python file"""
    # Import the config file as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep_config", config_path)
    if not spec or not spec.loader:
        raise click.BadParameter(f"Could not load {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Config should be in a variable called 'config'
    if not hasattr(module, 'config'):
        raise click.BadParameter(f"No 'config' variable found in {config_path}")
    
    return module.config

@click.group()
def cli():
    """Manage LLM training parameter sweeps"""
    pass

@cli.command()
@click.argument('config_path')
@click.option('--dry-run', is_flag=True, help="Show what would be submitted without submitting")
def submit(config_path: str, dry_run: bool):
    """Submit a new parameter sweep"""
    try:
        sweep_cfg = load_sweep_config(config_path)
        run_sweep(sweep_cfg, dry_run=dry_run)
    except Exception as e:
        console.print(f"[red]Error submitting sweep:[/red] {e}")
        sys.exit(1)

@cli.command()
@click.argument('sweep_dir')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table',
              help="Output format")
@click.option('--verbose', '-v', is_flag=True, help="Show detailed information including errors")
def status(sweep_dir: str, output_format: str, verbose: bool):
    """Show status of sweep jobs"""
    try:
        jobs = get_sweep_status(sweep_dir)
        
        if output_format == 'table':
            table = Table(title=f"Sweep Status: {sweep_dir}")
            table.add_column("Job")
            table.add_column("Status")
            table.add_column("Metric")
            if verbose:
                table.add_column("Error")
            
            for job in jobs.values():
                row = [
                    job.name,
                    f"[green]{job.status}[/green]" if job.status == "COMPLETED"
                    else f"[red]{job.status}[/red]" if job.status == "FAILED"
                    else f"[yellow]{job.status}[/yellow]",
                    f"{job.last_metric:.4f}" if job.last_metric else "-"
                ]
                if verbose and job.error:
                    row.append(job.error)
                elif verbose:
                    row.append("-")
                table.add_row(*row)
            
            console.print(table)
            
        else:  # json
            click.echo(json.dumps({name: job.__dict__ for name, job in jobs.items()}, 
                                indent=2))
            
    except Exception as e:
        console.print(f"[red]Error getting status:[/red] {e}")
        sys.exit(1)

@cli.command()
@click.argument('sweep_dir')
def cancel(sweep_dir: str):
    """Cancel all running jobs in sweep"""
    try:
        jobs = get_sweep_status(sweep_dir)
        running = [j.job_id for j in jobs.values() 
                  if j.job_id and j.status in {"PENDING", "RUNNING"}]
        
        if not running:
            console.print("No running jobs found")
            return
        
        if click.confirm(f"Cancel {len(running)} jobs?"):
            cancel_sweep_jobs(sweep_dir)
            console.print(f"[green]Cancelled {len(running)} jobs[/green]")
            
    except Exception as e:
        console.print(f"[red]Error cancelling jobs:[/red] {e}")
        sys.exit(1)

@cli.command()
@click.argument('sweep_dir')
@click.option('--dry-run', is_flag=True, help="Show what would be retried without submitting")
def retry(sweep_dir: str, dry_run: bool):
    """Retry failed jobs in sweep"""
    try:
        retry_failed_jobs(sweep_dir, dry_run=dry_run)
    except Exception as e:
        console.print(f"[red]Error retrying jobs:[/red] {e}")
        sys.exit(1)

@cli.command()
@click.argument('sweep_dir')
@click.option('--output', '-o', type=click.Path(), help="Output file (CSV/pickle)")
def summarize(sweep_dir: str, output: Optional[str]):
    """Generate summary of sweep results"""
    try:
        df = summarize_sweep(sweep_dir, output=output)
        if not output:
            # Print summary to console
            console.print("\nSweep Summary:")
            console.print(df.describe())
            console.print("\nBest results:")
            console.print(df.sort_values("metric").head())
    except Exception as e:
        console.print(f"[red]Error generating summary:[/red] {e}")
        sys.exit(1)

@cli.command()
@click.argument('sweep_dir')
@click.option('--x', required=True, help="X-axis parameter")
@click.option('--y', required=True, help="Y-axis metric")
@click.option('--hue', help="Optional parameter for color coding")
@click.option('--output', '-o', type=click.Path(), help="Output file for plot")
def plot(sweep_dir: str, x: str, y: str, hue: Optional[str], output: Optional[str]):
    """Plot sweep results"""
    try:
        plot_sweep_results(sweep_dir, x=x, y=y, hue=hue, output=output)
        if not output:
            console.print("Plot displayed (close window to continue)")
    except Exception as e:
        console.print(f"[red]Error plotting results:[/red] {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()