#!/bin/bash
# Slurm job for one long training run on the ORCHID GPU cluster.
# For short tests just run python directly on gpuhost001; use this for runs that
# should survive closing VS Code.
#
#     mkdir -p logs
#     sbatch setup/jasmin/train_gpu.sbatch                 # run_name "cnn", default init
#     sbatch setup/jasmin/train_gpu.sbatch cnn_s1 1        # run_name, init_seed
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

RUN_NAME="${1:-cnn}"             # 1st argument, default "cnn"
SEED_FLAG=""
[ -n "$2" ] && SEED_FLAG="--init_seed $2"   # 2nd argument, optional

python -m src.train \
    --data_dir /gws/ssde/j25a/esaclim/klara/data_gee/data_france_2020_j \
    --model_name SmallCNN \
    --exp_name jasmin_france \
    --run_name "$RUN_NAME" \
    --epochs 600 $SEED_FLAG
