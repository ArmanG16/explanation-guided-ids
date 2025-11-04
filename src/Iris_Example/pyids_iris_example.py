# pyids_example.py
# Example of training and evaluating an interpretable decision set using pyIDS

import sys
import os

# Add the pyIDS directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyIDS")))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from pyids.algorithms.ids import IDS
from pyids.data_structures.ids_rule import IDSRule
from pyids.algorithms.ids_classifier import mine_CARs
from pyarc.qcba.data_structures import QuantitativeDataFrame

# -----------------------------
# Load the iris dataset
# -----------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="class")

# Combine features and target into a single DataFrame
data = X.copy()
data["class"] = y

# Split into train/test sets
train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)

# -----------------------------
# Mine Class Association Rules (CARs)
# -----------------------------
cars = mine_CARs(
    df=train_df,
    rule_cutoff=20,
    sample=False,
    random_seed=42
)

print("\n--- Mined Cars ---")
print(f"\nMined CARs Count: {len(cars)}")
for i, car in enumerate(cars, 1):
    antecedent = getattr(car, "antecedent", None)
    consequent = getattr(car, "consequent", None)
    support = getattr(car, "support", None)
    confidence = getattr(car, "confidence", None)

    print(f"\nCAR {i}:")
    print(f"  IF {antecedent} THEN class = {consequent}")
    print(f"  Support: {support}, Confidence: {confidence}")

# -----------------------------
# Convert to QuantitativeDataFrame (required by IDS)
# -----------------------------
quant_train = QuantitativeDataFrame(train_df)
quant_test = QuantitativeDataFrame(test_df)

# -----------------------------
# Initialize IDS
# -----------------------------
ids_model = IDS(algorithm="RUSM")  # Optimizer: SLS, DLS, DUSM, RUSM

# -----------------------------
# Fit IDS model
# -----------------------------
ids_model.fit(
    quant_dataframe=quant_train,
    class_association_rules=cars,
    lambda_array=[1]*7,                 # optional, default is [1]*7
    default_class="majority_class_in_uncovered",
    random_seed=42
)

# -----------------------------
# Print learned rules
# -----------------------------
print("\n--- Learned Decision Rules ---")
print(f"\nDecision Rules Count: {len(ids_model.clf.rules)}")
for r in ids_model.clf.rules:
    print(IDSRule(rule=r).to_str())

# -----------------------------
# Evaluate on test data
# -----------------------------
y_pred = ids_model.predict(quant_test)
y_true = test_df["class"].astype(str)  # must match QuantitativeDataFrame string dtype

acc = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {acc:.3f}")
