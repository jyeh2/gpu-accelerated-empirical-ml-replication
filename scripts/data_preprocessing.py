# ============================================
# Section 2 — Data Preprocessing
# ============================================

import os
import pandas as pd
from scipy import sparse

cleaned_data_path = "../data/cleaned_data/"
model_data_path = "../data/modeling_data/"


def save_model_data(df, file_name):
    os.makedirs(model_data_path, exist_ok=True)
    df.to_csv(os.path.join(model_data_path, file_name), index=False)


# -------------------------
# Online Retail → Customer features + label
# -------------------------
online_retail_data = pd.read_csv(
    os.path.join(cleaned_data_path, "online_retail_data_clean.csv")
)

# Feature engineering
online_retail_data["TotalPrice"] = (
    online_retail_data["Quantity"] * online_retail_data["UnitPrice"]
)

online_retail_data["InvoiceDate"] = pd.to_datetime(
    online_retail_data["InvoiceDate"], errors="coerce"
)
online_retail_data["InvoiceHour"] = online_retail_data["InvoiceDate"].dt.hour
online_retail_data["InvoiceDay"] = online_retail_data["InvoiceDate"].dt.day
online_retail_data["InvoiceMonth"] = online_retail_data["InvoiceDate"].dt.month
online_retail_data["InvoiceDow"] = online_retail_data["InvoiceDate"].dt.dayofweek

online_retail_data["IsCancelled"] = (
    online_retail_data["InvoiceNo"].astype(str).str.startswith("C")
)

# Split timeline into feature period (first 70%) and target period (last 30%)
max_date = online_retail_data["InvoiceDate"].max()
min_date = online_retail_data["InvoiceDate"].min()
total_days = (max_date - min_date).days

feature_cutoff = min_date + pd.Timedelta(days=int(total_days * 0.7))

print(f"Feature period: {min_date.date()} to {feature_cutoff.date()}")
print(f"Target period: {feature_cutoff.date()} to {max_date.date()}")

feature_data = online_retail_data[
    online_retail_data["InvoiceDate"] <= feature_cutoff
]
target_data = online_retail_data[
    online_retail_data["InvoiceDate"] > feature_cutoff
]

print(f"\nFeature data: {len(feature_data)} rows")
print(f"Target data: {len(target_data)} rows")

# Group by customer for features (past)
g_features = feature_data.groupby("CustomerID")
customer_features = pd.DataFrame(
    {
        "CustomerID": g_features["CustomerID"].first().index,
        "Recency": (feature_cutoff - g_features["InvoiceDate"].max()).dt.days,
        "Frequency": g_features["InvoiceNo"].nunique(),
        "UniqueItems": g_features["Description"].nunique(),
        "AvgBasketQty": g_features["Quantity"].sum()
        / g_features["InvoiceNo"].nunique(),
        "AvgUnitPrice": g_features["UnitPrice"].mean(),
        "CancelRatio": g_features["InvoiceNo"].apply(
            lambda x: x.astype(str).str.startswith("C").mean()
        ),
    }
).reset_index(drop=True)

print(f"\nCustomers with features: {len(customer_features)}")

# Group by customer for target (future)
g_target = target_data.groupby("CustomerID")
lifetime_spend_future = g_target["TotalPrice"].sum()
spend_threshold = lifetime_spend_future.quantile(0.50)
big_spender = lifetime_spend_future >= spend_threshold

avg_interpurchase_future = g_target["InvoiceDate"].apply(
    lambda x: x.sort_values().diff().dt.days.mean()
)
engaged = avg_interpurchase_future < 4

label = (big_spender & engaged).astype(int)
label_df = pd.DataFrame(
    {"CustomerID": label.index, "HighValueCustomer": label.values}
)

print(f"Customers with labels: {len(label_df)}")

# Merge features + label, and drop customers without labels
customer_features = customer_features.merge(label_df, on="CustomerID", how="inner")

print(f"\nFinal dataset: {len(customer_features)} customers")
print("\nLabel distribution:")
print(
    customer_features["HighValueCustomer"].value_counts(normalize=True)
)

print(
    f" Features calculated from: {min_date.date()} to {feature_cutoff.date()}"
)
print(
    f" Target calculated from: {feature_cutoff.date()} to {max_date.date()}"
)
print(" These periods do NOT overlap!")

# Save full feature+label table for inspection
save_model_data(customer_features, "customer_features_model.csv")

# (Optionally define X,y here as in notebook)
X_customer = customer_features.drop(
    ["CustomerID", "HighValueCustomer"], axis=1
)
y_customer = customer_features["HighValueCustomer"]

# -------------------------
# Wine dataset merge
# -------------------------
red = pd.read_csv(os.path.join(cleaned_data_path, "wine_red_data_clean.csv"))
white = pd.read_csv(
    os.path.join(cleaned_data_path, "wine_white_data_clean.csv")
)

# Add binary label: 1 = red, 0 = white
red["is_red"] = 1
white["is_red"] = 0

# Merge into one unified dataset
wine_combined = pd.concat([red, white], ignore_index=True)

# Save combined modeling dataset
save_model_data(wine_combined, "wine_combined_data_model.csv")

# -------------------------
# CUP98 preprocessing
# -------------------------
# Explanation text in notebook describes leakage and why TARGET_D is dropped.
cup98 = pd.read_csv(os.path.join(cleaned_data_path, "cup98_data_clean.csv"))

categorical_cols = cup98.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(cup98, columns=categorical_cols, drop_first=True)

numeric_df = df_encoded.select_dtypes(include=["number"])
numeric_df = numeric_df.fillna(numeric_df.mean())

# Correlation with TARGET_B (for illustration / leakage inspection)
correlation_matrix = numeric_df.corr()
target_correlation = correlation_matrix["TARGET_B"].sort_values(
    ascending=False
)


# Drop explicit leakage feature(s) (TARGET_D; only if present)
drop_cols = ["TARGET_D"]
drop_cols = [c for c in drop_cols if c in cup98.columns]
cup98 = cup98.drop(columns=drop_cols)

# Ensure TARGET_B exists
if "TARGET_B" not in cup98.columns:
    raise ValueError("TARGET_B not found in CUP98 dataset.")

# Separate labels
y_cup = cup98["TARGET_B"].astype(int)
labels_df = pd.DataFrame({"TARGET_B": y_cup})
labels_df.to_csv(
    os.path.join(model_data_path, "cup98_labels.csv"),
    index=False,
    header=True,
)

# Base features (before any further selection)
X_base = cup98.drop(columns=["TARGET_B"])

# Variance threshold on numeric features (not fully visible in PDF; reconstruct)
from sklearn.feature_selection import VarianceThreshold

numeric_cols = X_base.select_dtypes(include=["number"]).columns
X_numeric = X_base[numeric_cols].fillna(X_base[numeric_cols].mean())

selector = VarianceThreshold(threshold=0.0)
X_var = selector.fit_transform(X_numeric)

# Save as sparse matrix for modeling
os.makedirs(model_data_path, exist_ok=True)
sparse.save_npz(
    os.path.join(model_data_path, "cup98_features.npz"),
    sparse.csr_matrix(X_var),
)
