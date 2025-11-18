import sys
import glob
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))

data_dir = os.path.join(BASE_DIR, "data/processed/bethdataset")
cars_dir = os.path.join(BASE_DIR, "data/cars/bethdataset.csv")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.Mining_Cars_Func import Mine_Cars
from src.utils.CSV_Files_To_DataFrame import CSV_to_DF

from pyids.algorithms.ids import IDS
import pandas as pd

from pyarc.qcba.data_structures import QuantitativeDataFrame

def Beth_Train(max_rows):

    cars = Mine_Cars(data_dir, max_rows, cars_dir)

    lambda_array = [1, 1, 1, 1, 1, 1, 1]

    df = CSV_to_DF(data_dir, max_rows=max_rows)
    df['class'] = df['class'].astype(str)

    quant_dataframe = QuantitativeDataFrame(df)

    ids = IDS(algorithm="SLS")
    ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

    print("\n--- Learned Decision Rules ---")
    print(f"Total Rules Selected by IDS: {len(ids.clf.rules)}\n")

    print("\n--- Simplified IDS Rule Summaries ---")

    for i, rule in enumerate(ids.clf.rules, start=1):
        car = rule.car

        antecedent = dict(car.antecedent)
        consequent = car.consequent
        confidence = car.confidence
        support = car.support
        f1 = getattr(rule, "f1", None)

        print(f"Rule {i}:")
        print(f"  IF {antecedent}")
        print(f"  THEN {consequent}")
        print(f"  Support: {support:.3f}, Confidence: {confidence:.3f}", end="")
        if f1 is not None:
            print(f", F1: {f1:.3f}")
        else:
            print()


    acc = ids.score(quant_dataframe) # accuracy is the percentage of the dataset covered by the generated rules
    print("\nAccuracy Score: " + str(acc))

if __name__ == "__main__":
    Beth_Train(10000)