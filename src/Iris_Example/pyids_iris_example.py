import sys
import os

# Add the pyIDS directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyIDS")))

import pandas as pd
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.data_structures.ids_rule import IDSRule

from pyarc.qcba.data_structures import QuantitativeDataFrame
import io
import requests

url = "https://raw.githubusercontent.com/kliegr/arcBench/master/data/folds_discr/train/iris0.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))

cars = mine_CARs(df, rule_cutoff=50)
print("\n--- Learned Decision Rules ---")
print(f"Mined CARs Count: {len(cars)}\n")
lambda_array = [1, 1, 1, 1, 1, 1, 1]

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


acc = ids.score(quant_dataframe)
print(f"Accuracy Score: {acc}\n")