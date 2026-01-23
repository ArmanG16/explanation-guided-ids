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

def fmax(lambda_dict):
    ids = IDS(galgorithm)
    ids.fit(class_association_rules=gcars, quant_dataframe=gquant_df, lambda_array=list(lambda_dict.values()))
    auc = ids.score_auc(gquant_df)
    MyPrint("Optimizing_Lambdas", "AUC: " + str(auc) + " for lambdas: " + str(lambda_dict))
    return auc

def Optimize_Lambdas(algorithm, cars, data_dir, max_rows, output_path, precision, iterations):
    global galgorithm, gquant_df, gcars
    galgorithm = algorithm
    gcars = cars
    df = CSV_to_DF(data_dir, max_rows=max_rows)
    df["class"] = df["class"].astype(str)
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

    best_lambdas = cord_asc.fit()
    best_lambdas.to_csv(output_path, index=False)
    return best_lambdas
