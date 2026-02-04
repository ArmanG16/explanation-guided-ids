#!/bin/bash
#SBATCH -N 1          # 
#SBATCH -n 8          # 
#SBATCH --mem=8g      # 
#SBATCH -J "Train on Titanic - Oversampling"  # 
#SBATCH -p short      # 
#SBATCH -t 12:00:00   # 

echo "Training on Titanic, Oversampling Based..."
source /home/mrcloutier/env/bin/activate
python -u Titanic_Oversampling.py