#!/bin/bash
#SBATCH --job-name=operational_train_weights
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Retrain one config's weights (only needed if models/ is lost — the repo ships them).
# Needs the historical feature matrices; set VERIFICATION_ROOT if they are not at
# the default location in pipeline/train_weights.py.
#
# Submit from the repo root, with your Python environment activated:
#   mkdir -p slurm/logs
#   sbatch --job-name=train_gefs_reduced slurm/submit_train_weights.sh gefs_reduced
# (Adjust --partition/--mem to your cluster.)

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/submit_train_weights.sh <config>}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
echo "=== Training weights: $CONFIG ===" && echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-$(hostname)}  Start: $(date)"
python -m pipeline.train_weights "$CONFIG"
echo "=== Done: $(date) ==="
