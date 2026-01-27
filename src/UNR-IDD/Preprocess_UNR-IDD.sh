#!/bin/bash
#SBATCH -N 1          # 
#SBATCH -n 2          # 
#SBATCH --mem=8g      # 
#SBATCH -J "Preprocessing UNR-IDD"  # 
#SBATCH -p short      # 
#SBATCH -t 12:00:00   # 

echo "Preprocessing UNR-IDD dataset..."
source /home/mrcloutier/env/bin/activate
python Preprocess_UNR-IDD.py