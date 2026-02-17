import os
import multiprocessing as mp

def Worker_Count():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
