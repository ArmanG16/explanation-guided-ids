#!/bin/bash
#SBATCH -N 1          # 
#SBATCH -n 8          # 
#SBATCH --mem=8g      # 
#SBATCH -J "Train on UNR-IDD"  # 
#SBATCH -p short      # 
#SBATCH -t 12:00:00   # 

echo "Training on UNR-IDD..."
source /home/csgilbert/env/bin/activate
python -u Train_UNR-IDD.py