#!/bin/bash
#SBATCH -N 1          # 
#SBATCH -n 2          # 
#SBATCH --cpus-per-task=16
#SBATCH --mem=8g      # 
#SBATCH -J "Train on UNR-IDD"  # 
#SBATCH -p short      # 
#SBATCH -t 12:00:00   # 

echo "Training on UNR-IDD..."
source /home/$(whoami)/env/bin/activate
python -u Train_UNR-IDD.py