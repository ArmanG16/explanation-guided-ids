import sys
import os
from src.utils.Print_Helper import MyPrint
from src.utils.CSV_Files_To_DataFrame import CSV_to_DF
import pandas as pd
from pyarc.qcba.data_structures import QuantitativeDataFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pyids.algorithms.ids import IDS

def Train(algorithm, lambda_array, cars, df, output_path):
    df["class"] = df["class"].astype(str)
    quant_dataframe = QuantitativeDataFrame(df)

    MyPrint("Training_Func", "Beginning training with pyIDS...")
    ids = IDS(algorithm=algorithm)
    ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

    MyPrint("Training_Func", f"Total Rules Selected by IDS: {len(ids.clf.rules)}\n")

    acc = ids.score(quant_dataframe) # accuracy is the percentage of the dataset covered by the generated rules

    rules_list = []

    for i, rule in enumerate(ids.clf.rules, start=1):
        car = rule.car

        antecedent = dict(car.antecedent)
        consequent = car.consequent
        confidence = car.confidence
        support = car.support
        f1 = getattr(rule, "f1", None)

        rules_list.append({
            "Rule_Index": i,
            "Antecedent": str(antecedent),
            "Consequent": str(consequent),
            "Support": support,
            "Confidence": confidence,
            "F1": f1,
            "Accuracy": acc if i == 1 else ""
        })

    rules_df = pd.DataFrame(rules_list)

    rules_df.to_csv(output_path, index=False)
