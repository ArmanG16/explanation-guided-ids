# learning_curve_ids_mlm_style.py
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Make sure pyIDS is importable (adjust path if needed) ----
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyIDS")))

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyarc.qcba.data_structures import QuantitativeDataFrame

from sklearn.model_selection import StratifiedShuffleSplit

# -------------------- Config --------------------
FILE_PATH   = "pyIDS/data/titanic0.csv"   # your local, discretized CSV
RULE_CUTOFF = 50                          # number of CARs to mine
LAMBDA      = [1, 1, 1, 1, 1, 1, 1]
VAL_SIZE    = 0.2                         # 20% validation split per repeat
TRAIN_FRACTIONS = np.linspace(0.1, 1.0, 10)  # 10%, 20%, ..., 100%
N_REPEATS   = 5                           # average over multiple random splits
RANDOM_STATE = 42

def train_and_score(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Fit IDS on train_df; return (train_acc, val_acc)."""
    cars = mine_CARs(train_df, rule_cutoff=RULE_CUTOFF)
    if not cars:   # defensive: can happen on tiny samples
        return np.nan, np.nan

    qtrain = QuantitativeDataFrame(train_df)
    ids = IDS(algorithm="SLS")
    ids.fit(quant_dataframe=qtrain, class_association_rules=cars, lambda_array=LAMBDA)

    train_acc = ids.score(qtrain)
    qval = QuantitativeDataFrame(val_df)
    val_acc = ids.score(qval)
    return train_acc, val_acc

def stratified_subsample(df: pd.DataFrame, y_col: str, n_samples: int, seed: int):
    """Take a (roughly) stratified sample of exactly n_samples."""
    if n_samples >= len(df):
        return df.sample(frac=1.0, random_state=seed)

    rng = np.random.RandomState(seed)
    parts = []
    for cls, grp in df.groupby(y_col):
        take = int(round(len(grp) * n_samples / len(df)))
        take = min(max(take, 0), len(grp))
        if take > 0:
            parts.append(grp.sample(n=take, random_state=rng))
    out = pd.concat(parts) if parts else df.sample(n=n_samples, random_state=rng)
    if len(out) < n_samples:
        top_up = df.drop(out.index).sample(n=n_samples - len(out), random_state=rng)
        out = pd.concat([out, top_up])
    return out.sample(frac=1.0, random_state=rng)  # shuffle

def main():
    df = pd.read_csv(FILE_PATH)

    # arcBench-style CSVs usually have the label in the last column
    y_col = df.columns[-1]
    y = df[y_col].values

    train_means, train_stds = [], []
    val_means, val_stds = [], []

    for frac in TRAIN_FRACTIONS:
        train_scores = []
        val_scores = []

        sss = StratifiedShuffleSplit(
            n_splits=N_REPEATS, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )

        for rep, (idx_train_all, idx_val) in enumerate(sss.split(df, y)):
            df_train_all = df.iloc[idx_train_all].copy()
            df_val = df.iloc[idx_val].copy()

            n_target = max(10, int(round(frac * len(df))))           # requested size
            n_target = min(n_target, len(df_train_all))              # cap by pool
            df_train = stratified_subsample(df_train_all, y_col, n_target,
                                            seed=RANDOM_STATE + rep)

            tr_acc, va_acc = train_and_score(df_train, df_val)
            train_scores.append(tr_acc)
            val_scores.append(va_acc)
            print(f"[size={len(df_train):4d}] rep {rep+1}/{N_REPEATS} "
                  f"-> train={tr_acc:.3f}, val={va_acc:.3f}")

        train_scores = np.array(train_scores, float)
        val_scores   = np.array(val_scores, float)

        train_means.append(np.nanmean(train_scores))
        train_stds.append(np.nanstd(train_scores))
        val_means.append(np.nanmean(val_scores))
        val_stds.append(np.nanstd(val_scores))

    x = (TRAIN_FRACTIONS * len(df)).astype(int)

    # ---- Plot in the MachineLearningMastery style: train vs validation with bands ----
    plt.figure(figsize=(8, 5))
    plt.plot(x, train_means, marker='o', label='Train accuracy')
    plt.fill_between(x,
                     np.array(train_means) - np.array(train_stds),
                     np.array(train_means) + np.array(train_stds),
                     alpha=0.15)

    plt.plot(x, val_means, marker='s', label='Validation accuracy')
    plt.fill_between(x,
                     np.array(val_means) - np.array(val_stds),
                     np.array(val_means) + np.array(val_stds),
                     alpha=0.15)

    plt.title("Learning Curves (IDS on titanic0.csv)")
    plt.xlabel("Training set size (samples)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
