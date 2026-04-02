#!/bin/bash
#SBATCH --job-name=mlp_beth
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load python
source ~/explanation-guided-ids/src/ml_based/ids-env/bin/activate

# Go to repo root, not ml_based
cd ~/explanation-guided-ids

echo "Running model_train.py on $(hostname) at $(date)"

# Call the script via its path from repo root
python src/ml_based/model_train.py

echo "Done at $(date)"
