#!/bin/bash
# Run all evolutionary runs (N_RUNS MAP-Elites + N_RUNS DE) on the HPC as ONE job array.
# For 30 + 30 runs: set N_RUNS=30 below AND --array=0-59 (array size = 2 x N_RUNS).
# Each array task = one run, on its own node/cores, all in parallel.
#
# Usage, from the repo folder on the cluster:
#     mkdir -p logs
#     sbatch run_all.slurm
#     squeue -u $USER              # check progress
# Then, when all tasks are done:
#     python reevaluate.py --runs results/evolution --seeds 30
#     python analyze.py
#
# Adapt the lines marked ADAPT to your cluster (ask the HPC support / read its docs).

#SBATCH --job-name=evo-rewards
#SBATCH --array=0-19                 # 0..N_RUNS-1 = MAP-Elites, N_RUNS..2*N_RUNS-1 = DE  (must be 0-$((2*N_RUNS-1)))
#SBATCH --cpus-per-task=8            # cores per run
#SBATCH --mem=4G
#SBATCH --time=02:00:00              # one run takes ~20 min on 12 cores; 2 h is a safe limit
#SBATCH --output=logs/%x_%a.out
##SBATCH --partition=<name>          # ADAPT: uncomment and set if your cluster requires a partition

# ADAPT: load Python the way your cluster wants it, e.g.
# module load python/3.11
# source ~/venv-evo/bin/activate     # a virtualenv with: pip install -r requirements.txt

N_RUNS=10                            # runs per algorithm (10 or 30); keep --array consistent

if [ "$SLURM_ARRAY_TASK_ID" -lt "$N_RUNS" ]; then
    ALGO=mapelites
else
    ALGO=de
fi
RUN=$((SLURM_ARRAY_TASK_ID % N_RUNS))

python run_evolution.py --algo "$ALGO" --run "$RUN" --workers "$SLURM_CPUS_PER_TASK"
