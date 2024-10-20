# maester


## Running the code

```
bash scripts/slurm/slurm.sh
```

## Check status of job

```
slurm q
```

## Upload checkpoints

```
python scripts/convert_dcp_to_hf.py jobs/mistral-7b/checkpoints/ ../2024-v2-maester/hf/ --upload danish-foundation-models/munin-7b-2024-v2-maester --name step-400 --base 
mistralai/Mistral-7B-v0.1 
```

har jeg kørt (I en VM på en maskine med 128 GB RAM, ingen GPU'er behøves). Det laver 2024-v2-maester/hf/step-400 og uploader den til https://huggingface.co/danish-foundation-models/munin-7b-2024-v2-maester-step-400

Kør i environment:

```{bash}
# start interactive session
srun --account=project_465000670 --partition=dev-g --time=02:00:00 --nodes=1 --gpus-per-node=0 --mem=128G --pty bash

# start singularity container
CONTAINER="/project/project_465000670/pytorch_rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2.sif"
singularity run --bind "$PWD,/scratch/project_465000670/2024-v2-maester/hf" "$CONTAINER"

# activate python venv
source .venv/bin/activate

# push a single checkpoint: 
python scripts/convert_dcp_to_hf.py jobs/mistral-7b/checkpoints/ /scratch/project_465000670/2024-v2-maester/hf --upload danish-foundation-models/munin-7b-2024-v2-maester --name step-800 --base mistralai/Mistral-7B-v0.1 

# push a list of checkpoints:
names=("step-0" "step-800" "step-1600" "step-3200" "step-6400", "step-8000")

# Loop through each name and run the script
for name in "${names[@]}"
do
    command="python scripts/convert_dcp_to_hf.py jobs/mistral-7b/checkpoints/ /scratch/project_465000670/2024-v2-maester/hf --upload danish-foundation-models/munin-7b-2024-v2-maester --name ${name} --base mistralai/Mistral-7B-v0.1"
    echo $command
    $command
done
```



