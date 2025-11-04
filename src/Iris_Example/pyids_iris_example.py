# pyids_example.py
# Example of training and evaluating an interpretable decision set using pyIDS

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyIDS")))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from pyids.algorithms.ids import mine_CARs, IDS
from pyids.data_structures.ids_rule import IDSRule

# Load the iris datasetp
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="class")

# Convert to one combined DataFrame
data = X.copy()
data["class"] 
# Split train/test
train_df, test_df = train_test_split(data, test_size=0.3, random_state=42)

# Mine class association rules (CARs)
cars = mine_CARs(train_df, class_name='class', minsup=0.1, minconf=0.6)

# Initialize IDS model
ids_model = IDS(
    target_class=None,        # allow multiple classes
    maxlen=3,                 # max rule length
    max_rules=50,             # limit number of rules
    algorithm='random',       # random rule selection strategy
)

# Fit model
ids_model.fit(cars, train_df, class_name='class')

# Print resulting rules
print("\n--- Learned Decision Rules ---")
for r in ids_model.rules:
    print(IDSRule(rule=r).to_str())

# Evaluate on test data
y_true = test_df["class"]
y_pred = ids_model.predict(test_df)

acc = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {acc:.3f}")
