# combined_pipeline.py
# ============================================
# Section 1 — Data Cleaning
# ============================================

import os
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import tarfile

base = os.path.dirname(os.path.abspath(__file__))

# Paths
raw_data_path = os.path.join(base, "../data/raw_data/")
clean_data_path = os.path.join(base, "../data/cleaned_data/")

def save_clean_data(df, file_name):
    os.makedirs(clean_data_path, exist_ok=True)
    df.to_csv(os.path.join(clean_data_path, file_name), index=False)

def extract_tar_xz(tar_path, extract_to="../data/"):
       # directory of this script
    tar_path = os.path.join(base, tar_path)
    extract_to = os.path.join(base, extract_to)

    os.makedirs(extract_to, exist_ok=True)

    with tarfile.open(tar_path, "r:xz") as tar:
        tar.extractall(path=extract_to)

extract_tar_xz("../data/data.tar.xz")

# ---- Load raw datasets ----


wine_red_data_raw = pd.read_csv(
    os.path.join(raw_data_path, "winequality/winequality-red.csv")
)
wine_white_data_raw = pd.read_csv(
    os.path.join(raw_data_path, "winequality/winequality-white.csv")
)

online_retail_data_raw = pd.read_excel(
    os.path.join(raw_data_path, "retail/Online Retail.xlsx")
)

cup98_data_raw = pd.read_csv(
    os.path.join(raw_data_path, "cup98/cup98LRN.txt")
)

# ---- Online Retail cleaning ----
# Keep only rows with CustomerID (we care about per-customer behavior)
online_retail_data = online_retail_data_raw.copy()
online_retail_data = online_retail_data.dropna(subset=["CustomerID"])
save_clean_data(online_retail_data, "online_retail_data_clean.csv")

# ---- Wine cleaning (they are already pretty clean; just save) ----
save_clean_data(wine_red_data_raw, "wine_red_data_clean.csv")
save_clean_data(wine_white_data_raw, "wine_white_data_clean.csv")


# ---- CUP98 cleaning & imputation ----
def impute_cup98(df):
    """
    Impute CUP98 according to:
    - numeric / mixed numeric → mean
    - categorical / boolean → treat missing-like tokens as "MISSING"
    """
    df = df.copy()
    for col in df.columns:
        # Try converting to numeric to detect mixed-type numeric/binary columns
        coerced = pd.to_numeric(df[col], errors="coerce")

        # Case A: Column *becomes* numeric after coercion → treat as numeric
        if coerced.notna().sum() > 0 and coerced.isna().sum() < len(df):
            df[col] = coerced
            df[col] = df[col].fillna(df[col].mean())
            continue

        # Case B: Column does NOT coerce to numeric → treat as categorical
        df[col] = df[col].astype(str).replace(
            ["", " ", "X", "nan", "NaN", "NAN"], "MISSING"
        )
        df[col] = df[col].fillna("MISSING")

    return df


# Drop columns with > 50% missing, then impute
cup98_data_raw = cup98_data_raw.loc[:, cup98_data_raw.isna().mean() < 0.50]
cup98_data_clean = impute_cup98(cup98_data_raw)
save_clean_data(cup98_data_clean, "cup98_data_clean.csv")

# Remove the huge raw_data directory, as in the notebook
import shutil

shutil.rmtree(raw_data_path)
print(f"Removed raw data directory: {raw_data_path}")