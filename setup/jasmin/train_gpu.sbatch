#!/bin/bash
# Slurm job for one long training run on the ORCHID GPU cluster.
# For short tests just run python directly on gpuhost001; use this for runs that
# should survive closing VS Code.
#
#     mkdir -p logs
#     sbatch setup/jasmin/train_gpu.sbatch
#     squeue -u $USER                 # is it queued / running?
#     tail -f logs/slurm-<jobid>.out
#
# If sbatch rejects the account/qos names, check
# https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/
#SBATCH --job-name=cci_train
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export WANDB_MODE=offline        # no account needed; `wandb sync wandb/offline-run-*` later if wanted

python -m src.train \
    --data_dir /gws/ssde/j25a/esaclim/klara/data_gee/data_france_2020_j \
    --model_name SmallCNN \
    --exp_name jasmin_france \
    --run_name cnn \
    --epochs 600
