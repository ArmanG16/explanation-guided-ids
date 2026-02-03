#!/bin/bash
#SBATCH -N 1          # 
#SBATCH -n 8          # 
#SBATCH --mem=8g      # 
#SBATCH -J "Train on Titanic"  # 
#SBATCH -p short      # 
#SBATCH -t 12:00:00   # 

echo "Training on Titanic..."
source /home/mrcloutier/env/bin/activate
python -u Train_Titanic_With_Lambdas.py