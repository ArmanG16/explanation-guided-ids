import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Config ----------------
DATA_PATH = "data/processed/bethdataset/"
OUTPUT_PATH = "data/results/mlp_results.csv"
LABEL_CANDIDATES = ["sus", "evil", "label", "attack", "target"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 10  # keep short for testing; increase once balanced
BATCH_SIZE = 64

# ---------------- Helper Functions ----------------
def detect_label_column(df):
    for col in df.columns:
        if col.lower() in LABEL_CANDIDATES:
            return col
    return df.columns[-1]

def load_and_combine_csvs(data_path):
    files = glob.glob(os.path.join(data_path, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        label_col = detect_label_column(df)
        if label_col.lower() == "hostname":
            continue
        dfs.append(df)
    if not dfs:
        raise ValueError("No valid CSV files found.")
    return pd.concat(dfs, ignore_index=True)

# ---------------- Load and Prepare Data ----------------
print("Loading processed datasets...")
df = load_and_combine_csvs(DATA_PATH)
label_col = detect_label_column(df)
print(f"Detected label column: {label_col}")

X = df.drop(columns=[label_col])
y = df[label_col].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------- Handle Class Imbalance ----------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights}")

# ---------------- Build MLP Model ----------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# ---------------- Train Model ----------------
print("Training MLP model with class weighting...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    verbose=1
)

# ---------------- Evaluate Model ----------------
print("Evaluating model...")
test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
f1_score = 2 * ((test_prec * test_rec) / (test_prec + test_rec + 1e-7))

results = {
    "Test Accuracy": test_acc,
    "Precision": test_prec,
    "Recall": test_rec,
    "F1 Score": f1_score
}
print("\nModel Performance:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# ---------------- Save Results ----------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
pd.DataFrame([results]).to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved model metrics to {OUTPUT_PATH}")

# ---------------- Save Model ----------------
model.save("data/results/mlp_model_balanced.h5")
print("Saved trained model to data/results/mlp_model_balanced.h5")
