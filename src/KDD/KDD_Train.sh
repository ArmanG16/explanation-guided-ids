#!/bin/bash
#SBATCH -N 1          # 
#SBATCH -n 2          # 
#SBATCH --mem=8g      # 
#SBATCH -J "Training on NSL-KDD"  # 
#SBATCH -p short      # 
#SBATCH -t 12:00:00   # 

echo "Training on NSL-KDD dataset..."
source /home/mrcloutier/env/bin/activate
python -u KDD_Train.py