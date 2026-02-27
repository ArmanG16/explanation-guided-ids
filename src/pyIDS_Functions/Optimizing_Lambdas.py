import sys
import os
from src.utils.Print_Helper import MyPrint
from src.utils.Multithreading_Helper import Worker_Count
import pandas as pd
from pyarc.qcba.data_structures import QuantitativeDataFrame
import multiprocessing as mp
import itertools
import random


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyids.model_selection.random_search import RandomSearch
from pyids.model_selection.grid_search import GridSearch

from pyarc.qcba.data_structures import QuantitativeDataFrame

galgorithm = None
gquant_df = None
gcars = None
gauc_list = None
cord_asc_individual = None

worker_gcars = None
worker_gquant_df = None
worker_galgorithm = None

def init_worker(cars, quant_df, algorithm):
    global worker_gcars, worker_gquant_df, worker_galgorithm
    worker_gcars = cars
    worker_gquant_df = quant_df
    worker_galgorithm = algorithm

def fmax_individual(lambda_dict):  
    ids = IDS(worker_galgorithm)
    ids.fit(class_association_rules=worker_gcars, quant_dataframe=worker_gquant_df, lambda_array=list(lambda_dict.values()))
    auc = ids.score_auc(worker_gquant_df)
    MyPrint("Optimizing_Lambdas", "Individual, AUC: " + str(auc) + " for lambdas: " + str(lambda_dict))
    return auc

def fmax_net(lambda_dict):
    global gcars, gquant_df, galgorithm, gauc_list
    ids = IDS(galgorithm)
    ids.fit(class_association_rules=gcars, quant_dataframe=gquant_df, lambda_array=list(lambda_dict.values()))
    auc = ids.score_auc(gquant_df)
    MyPrint("Optimizing_Lambdas", "Net, AUC: " + str(auc) + " for lambdas: " + str(lambda_dict))
    return auc

def fit_lambda(arg_name):
    return arg_name, cord_asc_individual.fit_1lambda(arg_name)

def Optimize_Lambdas(algorithm, cars, df, output_path, individiual_iterations=3, individual_precision=50, iterations=3, precision=50, search_type="coordinate", grid_step=200):
    MyPrint("Optimizing_Lambdas", "Starting lambda optimization...")
    global galgorithm, gquant_df, gcars
    galgorithm = algorithm
    gcars = cars
    gquant_df = QuantitativeDataFrame(df)

    func_args_ranges=dict(
    l1=(1, 1000),
    l2=(1, 1000),
    l3=(1, 1000),
    l4=(1, 1000),
    l5=(1, 1000),
    l6=(1, 1000),
    l7=(1, 1000)
    )

    if (search_type == "coordinate"):
        MyPrint("Optimizing_Lambdas", "Using coordinate ascent for optimization with precision " + str(precision) + " and iterations " + str(iterations) + "...")
        
        global cord_asc_individual
        cord_asc_individual = CoordinateAscent(
            func=fmax_individual,
            func_args_ranges=func_args_ranges,
            ternary_search_precision=individual_precision,
            max_iterations=individiual_iterations
        )

        best_lambdas_initial = {}
        lambda_names = list(func_args_ranges.keys())
        
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))

        MyPrint("Optimizing_Lambdas", "Found " + str(num_workers) + " workers for parallel optimization.")

        mp.set_start_method("fork", force=True)

        #Prepare OS
        old_env = {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        }

        #Limit threads for each process
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(cars, gquant_df, algorithm)
        ) as pool:

            results = pool.map(fit_lambda, lambda_names)

        for arg_name, best_val in results:
            best_lambdas_initial[arg_name] = best_val

        MyPrint("Optimizing_Lambdas", f"Best individual lambdas: {best_lambdas_initial}", success=True)

        #Bring back old environment
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        search_radius = 100

        func_args_ranges_net = {
            k: (max(1, int(v - search_radius)), int(v + search_radius))
            for k, v in best_lambdas_initial.items()
        }

        cord_asc_net = CoordinateAscent(
            func=fmax_net,
            func_args_ranges=func_args_ranges_net,
            ternary_search_precision=precision,
            max_iterations=iterations
        )

        best_lambdas = cord_asc_net.fit()

        return best_lambdas
    
    elif (search_type == "grid"):
        MyPrint("Optimizing_Lambdas", "Using grid search for optimization")
        
        param_grid = {
            k: range(v[0], v[1] + 1, grid_step)
            for k, v in func_args_ranges.items()
        }       

        grid = GridSearch(
            func=fmax_net,
            func_args_spaces=param_grid,
            max_iterations=iterations
        )

        best_lambdas = grid.fit()


        return best_lambdas
    
    elif (search_type == "random"):
        MyPrint("Optimizing_Lambdas", "Using random search for optimization with " + str(iterations) + " samples")

        random_search = RandomSearch(
            func=fmax_net,
            func_args_ranges=func_args_ranges,
            max_iterations=iterations
        )
        best_lambdas = random_search.fit()

    else:
        return [1, 1, 1, 1, 1, 1, 1]
