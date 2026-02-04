import sys
import os
from src.utils.Print_Helper import MyPrint
from src.utils.CSV_Files_To_DataFrame import CSV_to_DF
import pandas as pd
from pyarc.qcba.data_structures import QuantitativeDataFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent

from pyarc.qcba.data_structures import QuantitativeDataFrame

galgorithm = None
gquant_df = None
gcars = None
glambdas = None
gauc_list = None

def fmax(lambda_dict):
    global glambdas, gcars, gquant_df, galgorithm, gauc_list
    ids = IDS(galgorithm)
    ids.fit(class_association_rules=gcars, quant_dataframe=gquant_df, lambda_array=list(lambda_dict.values()))
    auc = ids.score_auc(gquant_df)
    if glambdas is None:
        glambdas = [lambda_dict.copy()]
        gauc_list = [auc]
    else:
        glambdas.append(lambda_dict.copy())
        gauc_list.append(auc)
    MyPrint("Optimizing_Lambdas", "AUC: " + str(auc) + " for lambdas: " + str(lambda_dict))
    return auc

def Optimize_Lambdas(algorithm, cars, df, output_path, precision, iterations):
    MyPrint("Optimizing_Lambdas", "Starting lambda optimization...")
    global galgorithm, gquant_df, gcars
    galgorithm = algorithm
    gcars = cars
    gquant_df = QuantitativeDataFrame(df)
    cord_asc = CoordinateAscent(
        func=fmax,
        func_args_ranges=dict(
            l1=(1, 1000),
            l2=(1, 1000),
            l3=(1, 1000),
            l4=(1, 1000),
            l5=(1, 1000),
            l6=(1, 1000),
            l7=(1, 1000)
        ),
        ternary_search_precision=precision,
        max_iterations=iterations
    )
    global glambdas
    best_lambdas = cord_asc.fit()

    df_to_save = pd.DataFrame(glambdas)
    df_to_save.insert(0, "AUC", gauc_list)
    df_to_save.to_csv(output_path, index=False)

    return best_lambdas
