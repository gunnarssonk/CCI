#!/bin/bash
# Slurm job for one v7 training run on the ORCHID GPU cluster.
# Same resources as the v6 job script (setup/jasmin/train_gpu.sbatch); 24 hours is the cluster maximum.
# Everything after the script name is the python module and its flags:
#
#     mkdir -p logs/v7
#     sbatch setup/jasmin/v7/train_gpu.sbatch src.v7.experiments train --setup all6 --model cnn
#     sbatch setup/jasmin/v7/train_gpu.sbatch src.v7.experiments train --setup nonl --model pw
#     python -m src.v7.experiments submit --after <build job>     # queues all of them, with their scoring jobs
#     squeue -u $USER
#     tail -f logs/v7/slurm-<jobid>.out
#SBATCH --job-name=cci_v7_train
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v7/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export WANDB_MODE=offline
export WANDB_DIR=logs/v7
export PYTHONUNBUFFERED=1
echo "job $SLURM_JOB_ID on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "command: python -m $*"
echo "git commit: $(git rev-parse --short HEAD)"
echo
python -m "$@"
