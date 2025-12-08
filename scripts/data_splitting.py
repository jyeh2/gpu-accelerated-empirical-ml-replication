# ============================================
# Section 3 — Data Splitting
# ============================================

import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split  # imported in notebook, rarely used

base = os.path.dirname(os.path.abspath(__file__))

model_data_path = os.path.join(base, "../data/modeling_data/")
splits_path     = os.path.join(base, "../data/splits/")


os.makedirs(os.path.join(splits_path, "7030"), exist_ok=True)
os.makedirs(os.path.join(splits_path, "5050"), exist_ok=True)
os.makedirs(os.path.join(splits_path, "3070"), exist_ok=True)


def save_split(X_train, X_test, y_train, y_test, data, folder):
    """
    Save X_train, X_test, y_train, y_test for a given dataset + folder
    Exactly mirrors notebook logic: sparse matrices saved as .npz, others as .csv
    """
    out_path = os.path.join(splits_path, folder)
    os.makedirs(out_path, exist_ok=True)

    # ---- Save X_train ----
    if sparse.issparse(X_train):
        sparse.save_npz(
            os.path.join(out_path, f"X_train_{data}.npz"), X_train
        )
    else:
        pd.DataFrame(X_train).to_csv(
            os.path.join(out_path, f"X_train_{data}.csv"), index=False
        )

    # ---- Save X_test ----
    if sparse.issparse(X_test):
        sparse.save_npz(
            os.path.join(out_path, f"X_test_{data}.npz"), X_test
        )
    else:
        pd.DataFrame(X_test).to_csv(
            os.path.join(out_path, f"X_test_{data}.csv"), index=False
        )

    # ---- Save y labels ----
    y_train.to_csv(
        os.path.join(out_path, f"y_train_{data}.csv"), index=False
    )
    y_test.to_csv(
        os.path.join(out_path, f"y_test_{data}.csv"), index=False
    )

    print(f"Saved split to {out_path}")


def manual_stratified_split(X, y, train_ratio, seed=42):
    """
    Manual stratified split that retains indices (used in notebook
    to verify no leakage).
    """
    np.random.seed(seed)
    y_arr = np.array(y)

    train_idx = []
    test_idx = []

    # Stratified split
    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        np.random.shuffle(cls_idx)
        n_train = int(len(cls_idx) * train_ratio)

        train_idx.extend(cls_idx[:n_train])
        test_idx.extend(cls_idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # Slice, but DO NOT reset index yet
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        # e.g., numpy array or sparse matrix
        X_train = X[train_idx]
        X_test = X[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test, train_idx, test_idx


# -------------------------
# Splitting the Wine dataset
# -------------------------
wine = pd.read_csv(os.path.join(model_data_path, "wine_combined_data_model.csv"))
y = wine["is_red"]
X = wine.drop(columns=["is_red"])

print("Class distribution (0 = white, 1 = red):")
print(y.value_counts())
print("\nPercentage:")
print(y.value_counts(normalize=True))

# Not balanced → stratified sampling

# 70 / 30
X_train_7030, X_test_7030, y_train_7030, y_test_7030, idx_train_7030, idx_test_7030 = (
    manual_stratified_split(X, y, 0.70)
)
save_split(X_train_7030, X_test_7030, y_train_7030, y_test_7030, "wine", "7030")
print("7030 leakage:", len(set(idx_train_7030) & set(idx_test_7030)))

# 30 / 70
X_train_3070, X_test_3070, y_train_3070, y_test_3070, idx_train_3070, idx_test_3070 = (
    manual_stratified_split(X, y, 0.30)
)
save_split(X_train_3070, X_test_3070, y_train_3070, y_test_3070, "wine", "3070")
print("3070 leakage:", len(set(idx_train_3070) & set(idx_test_3070)))

# 50 / 50
X_train_5050, X_test_5050, y_train_5050, y_test_5050, idx_train_5050, idx_test_5050 = (
    manual_stratified_split(X, y, 0.50)
)
save_split(X_train_5050, X_test_5050, y_train_5050, y_test_5050, "wine", "5050")
print("5050 leakage:", len(set(idx_train_5050) & set(idx_test_5050)))

# -------------------------
# Splitting the Customer dataset
# -------------------------
customer = pd.read_csv(
    os.path.join(model_data_path, "customer_features_model.csv")
)

target_col = "HighValueCustomer"
y = customer[target_col]
X = customer.drop(columns=[target_col])

print("Counts:")
print(y.value_counts())
print("\nPercentages:")
print(y.value_counts(normalize=True))

# Still a bit imbalanced → stratified

# 70 / 30
(
    X_train_7030,
    X_test_7030,
    y_train_7030,
    y_test_7030,
    idx_train_7030,
    idx_test_7030,
) = manual_stratified_split(X, y, 0.70)
save_split(
    X_train_7030,
    X_test_7030,
    y_train_7030,
    y_test_7030,
    "customer",
    "7030",
)
print("Customer 7030 leakage:", len(set(idx_train_7030) & set(idx_test_7030)))

# 30 / 70
(
    X_train_3070,
    X_test_3070,
    y_train_3070,
    y_test_3070,
    idx_train_3070,
    idx_test_3070,
) = manual_stratified_split(X, y, 0.30)
save_split(
    X_train_3070,
    X_test_3070,
    y_train_3070,
    y_test_3070,
    "customer",
    "3070",
)
print("Customer 3070 leakage:", len(set(idx_train_3070) & set(idx_test_3070)))

# 50 / 50
(
    X_train_5050,
    X_test_5050,
    y_train_5050,
    y_test_5050,
    idx_train_5050,
    idx_test_5050,
) = manual_stratified_split(X, y, 0.50)
save_split(
    X_train_5050,
    X_test_5050,
    y_train_5050,
    y_test_5050,
    "customer",
    "5050",
)
print("Customer 5050 leakage:", len(set(idx_train_5050) & set(idx_test_5050)))

# -------------------------
# Splitting the CUP98 dataset
# -------------------------
X = sparse.load_npz(os.path.join(model_data_path, "cup98_features.npz"))
y = pd.read_csv(os.path.join(model_data_path, "cup98_labels.csv"))["TARGET_B"]

print("Counts:")
print(y.value_counts())
print("\nPercentages:")
print(y.value_counts(normalize=True))

# Worst distribution → stratify

# 70 / 30
(
    X_train_7030,
    X_test_7030,
    y_train_7030,
    y_test_7030,
    idx_train_7030,
    idx_test_7030,
) = manual_stratified_split(X, y, 0.70)
save_split(
    X_train_7030,
    X_test_7030,
    y_train_7030,
    y_test_7030,
    "cup98",
    "7030",
)
print("CUP98 7030 leakage:", len(set(idx_train_7030) & set(idx_test_7030)))

# 30 / 70
(
    X_train_3070,
    X_test_3070,
    y_train_3070,
    y_test_3070,
    idx_train_3070,
    idx_test_3070,
) = manual_stratified_split(X, y, 0.30)
save_split(
    X_train_3070,
    X_test_3070,
    y_train_3070,
    y_test_3070,
    "cup98",
    "3070",
)
print("CUP98 3070 leakage:", len(set(idx_train_3070) & set(idx_test_3070)))

# 50 / 50
(
    X_train_5050,
    X_test_5050,
    y_train_5050,
    y_test_5050,
    idx_train_5050,
    idx_test_5050,
) = manual_stratified_split(X, y, 0.50)
save_split(
    X_train_5050,
    X_test_5050,
    y_train_5050,
    y_test_5050,
    "cup98",
    "5050",
)
print("CUP98 5050 leakage:", len(set(idx_train_5050) & set(idx_test_5050)))

import shutil
shutil.rmtree(model_data_path)