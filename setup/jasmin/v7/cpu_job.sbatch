#!/bin/bash
# Slurm job for one v7 step that needs no GPU (CCI crops, Earth Engine blocks, tests).
# Everything after the script name is the python module and its flags:
#
#     mkdir -p logs/v7
#     sbatch setup/jasmin/v7/cpu_job.sbatch src.v7.crop_cci
#     sbatch --partition=debug --qos=debug --time=00:30:00 setup/jasmin/v7/cpu_job.sbatch src.v7.one_block_test --box netherlands --block r2_c3
#     squeue -u $USER
#     tail -f logs/v7/slurm-<jobid>.out
#
# The exact command is printed at the top of the log, and each script also writes
# a run_*.json with its command and settings next to its output.
#SBATCH --job-name=cci_v7
#SBATCH --partition=standard
#SBATCH --account=esaclim
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v7/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
echo "job $SLURM_JOB_ID on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "command: python -m $*"
echo "git commit: $(git rev-parse --short HEAD)"
echo
python -m "$@"
