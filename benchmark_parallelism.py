from dataclasses import dataclass
from typing import List, Dict
import itertools
import json
from pathlib import Path
import subprocess
from datetime import datetime

from pydantic import BaseModel

class BenchmarkConfig(BaseModel):
    """Configuration for a scaling benchmark run"""
    # Base configuration
    model_name: str = "llama"
    model_configs: Dict[str, int] = {  # model size -> max per-rank batch size
        "7b": 32,
        "13b": 16,
        "70b": 2,
    }
    
    # Parallelism configurations to test
    dp_shard_sizes: List[int] = [1, 8, 16, 32, 64]
    dp_replicate_sizes: List[int] = [1] 
    tp_sizes: List[int] = [1, 2, 4, 8]
    
    # SLURM configuration
    account: str
    partition: str
    time: str = "2:00:00"
    container: str
    
    # Training configuration  
    warmup_steps: int = 10
    benchmark_steps: int = 50
    log_freq: int = 5
    
    # Output configuration
    benchmark_dir: Path

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results_dir = config.benchmark_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.results_dir / "config.json", "w") as f:
            f.write(config.model_dump_json(indent=2))
    
    def run(self):
        """Generate configs and submit benchmark jobs"""
        groups = self._group_configs()
        print(f"Generated {len(groups)} job groups")
        
        for i, (num_nodes, configs) in enumerate(groups.items()):
            print(f"\nSubmitting job group {i+1}/{len(groups)} ({num_nodes} nodes)")
            print(f"Contains {len(configs)} configurations:")
            for c in configs:
                print(f"  {c['flavor']}: DP={c['data_parallel_shard_degree']}x{c['data_parallel_replicate_degree']}, TP={c['tensor_parallel_degree']}, batch={c['train_batch_size']}")
            
            job_dir = self._setup_job(num_nodes, configs)
            self._submit_job(job_dir)
    
    def _group_configs(self) -> Dict[int, List[Dict]]:
        """Generate and group configurations by node count"""
        groups: Dict[int, List[Dict]] = {}
        
        for model_size, dp_shard, dp_rep, tp in itertools.product(
            self.config.model_configs.keys(),
            self.config.dp_shard_sizes,
            self.config.dp_replicate_sizes, 
            self.config.tp_sizes
        ):
            total_ranks = dp_shard * dp_rep * tp
            if total_ranks > 2048:  # Skip too large configurations
                continue
                
            num_nodes = (total_ranks + 7) // 8  # 8 GPUs per node
            
            # Per-rank batch size scaled down for larger models
            base_batch = self.config.model_configs[model_size]
            
            config = {
                "model_name": self.config.model_name,
                "flavor": model_size,
                "train_batch_size": base_batch,
                "data_parallel_shard_degree": dp_shard,
                "data_parallel_replicate_degree": dp_rep,
                "tensor_parallel_degree": tp,
                "pipeline_parallel_degree": 1,
                "train_num_steps": self.config.warmup_steps + self.config.benchmark_steps,
                "log_freq": self.config.log_freq
            }
            
            if num_nodes not in groups:
                groups[num_nodes] = []
            groups[num_nodes].append(config)
        
        return groups
    
    def _setup_job(self, num_nodes: int, configs: List[Dict]) -> Path:
        """Create job directory and write necessary files"""
        job_dir = self.results_dir / f"nodes_{num_nodes}"
        job_dir.mkdir()
        
        # Write configs
        with open(job_dir / "configs.json", "w") as f:
            json.dump(configs, f, indent=2)
        
        # Write runner script
        with open(job_dir / "run_benchmarks.py", "w") as f:
            f.write("""
import json
import os
import sys
import time
from pathlib import Path

def main():
    with open("configs.json") as f:
        configs = json.load(f)
    
    for i, config in enumerate(configs, 1):
        print(f"\\nRunning configuration {i}/{len(configs)}")
        
        # Write temporary config
        with open("temp_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Run training
        os.system(f"python train.py --load_config temp_config.json")
        
        # Cleanup and delay
        os.unlink("temp_config.json")
        time.sleep(10)

if __name__ == "__main__":
    main()
""")
        
        # Write SLURM script
        from submit import SBATCH_TEMPLATE
        slurm_script = SBATCH_TEMPLATE.format(
            job_name=f"bench_{num_nodes}n",
            num_nodes=num_nodes,
            partition=self.config.partition,
            account=self.config.account,
            time=self.config.time,
            container=self.config.container
        )
        with open(job_dir / "slurm.sh", "w") as f:
            f.write(slurm_script)
            
        return job_dir
    
    def _submit_job(self, job_dir: Path):
        """Submit SLURM job"""
        subprocess.run(["sbatch", str(job_dir / "slurm.sh")], check=True)

def main():
    config = BenchmarkConfig(
        benchmark_dir=Path("benchmark_results"),
        account="project_465000954",
        partition="standard-g",
        container="/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.2.sif",
    )
    
    runner = BenchmarkRunner(config)
    runner.run()

if __name__ == "__main__":
    main()