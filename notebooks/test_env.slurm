#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --output=test_env.out
#SBATCH --error=test_env.err
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short

set -x  # echo commands as they run

echo "HOST: $(hostname)"
echo "WHOAMI: $(whoami)"
echo "PWD: $(pwd)"

# Show what python we have *without* any activation:
which python
python --version || echo "python failed"

# Now try your venv python explicitly
ls -l /home/nsewell/py312/bin/python || echo "venv python not found"

# If it exists, run a tiny test
/home/nsewell/py312/bin/python -c "import sys; print('VENV PY OK', sys.version)"
