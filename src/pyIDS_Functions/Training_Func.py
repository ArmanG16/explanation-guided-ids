import json
import sys
import os
from src.utils.Print_Helper import MyPrint
from src.utils.CSV_Files_To_DataFrame import CSV_to_DF
import pandas as pd
from pyarc.qcba.data_structures import QuantitativeDataFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pyids.algorithms.ids import IDS

def Train(algorithm, lambda_array, cars, df, output_path):
    quant_dataframe = QuantitativeDataFrame(df)

    MyPrint("Training_Func", "Beginning training with pyIDS...")
    ids = IDS(algorithm=algorithm)
    ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

    MyPrint("Training_Func", f"Total Rules Selected by IDS: {len(ids.clf.rules)}\n")

    acc = ids.score(quant_dataframe) # accuracy is the percentage of the dataset covered by the generated rules

    rules_list = []
    json_list = []

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

        readable_rule = f"IF {antecedent} THEN class = {consequent}"

        json_list.append({
            "row_index": i,
            "true_class": None,  # not known at training time
            "matched_rules": 1,
            "top_explanations": [
                {
                    "Predicted_Class": str(consequent),
                    "Confidence": confidence,
                    "Support": support,
                    "F1": f1,
                    "Readable_Rule": readable_rule
                }
            ]
        })

    rules_df = pd.DataFrame(rules_list)

    rules_df.to_csv(output_path, index=False)

    if json_output_path is None:
        json_output_path = output_path.replace(".csv", ".json")

    with open(json_output_path, "w") as f:
        json.dump(json_list, f, indent=2)

    MyPrint("Training_Func", f"JSON results saved to: {json_output_path}")
