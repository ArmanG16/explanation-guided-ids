#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -J "Train on USNW-NB15"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --mem=256g


echo "Preprocessing on USNW-NB15..."
source /home/mrcloutier/explanation-guided-ids/venv/bin/activate
python -u Preprocess_USNW.py

