"""
Week 1 Task: Small Python Data-Processing App
Loads, cleans, and processes raw cloud usage data using Python and NumPy.
Compatible with standard Python execution, Jupyter Notebooks, and Google Colab.
"""

from typing import Tuple, Dict, Any
import os
import pandas as pd
import numpy as np


def load_raw_dataset(filepath: str = "raw_dataset.csv") -> pd.DataFrame:
    """
    Loads raw dataset from a CSV file.
    Checks relative path, script directory, and tasks/Week1 candidate paths.
    If the file is not found (e.g., initial run in Google Colab), loads an in-memory sample.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        filepath,
        os.path.join(script_dir, filepath),
        os.path.join("tasks", "Week1", filepath),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[INFO] Loaded dataset from '{path}' with shape: {df.shape}")
            return df

    print(f"[WARN] File '{filepath}' not found in candidates. Loading built-in raw dataset...")
    raw_data = [
        {"customer_id": "C101", "service_type": "Cloud_Compute", "usage_hours": "120.5", "monthly_spend": "$245.80", "latency_ms": "32.4"},
        {"customer_id": "C102", "service_type": "Cloud_Storage", "usage_hours": "45.0", "monthly_spend": "$45.00", "latency_ms": "18.2"},
        {"customer_id": "C103", "service_type": "AI_Inference", "usage_hours": None, "monthly_spend": "$310.50", "latency_ms": "110.6"},
        {"customer_id": "C104", "service_type": "Cloud_Compute", "usage_hours": "300.2", "monthly_spend": "$590.00", "latency_ms": None},
        {"customer_id": "C105", "service_type": "Database", "usage_hours": "80.0", "monthly_spend": "$120.00", "latency_ms": "24.1"},
        {"customer_id": "C106", "service_type": "AI_Inference", "usage_hours": "-15.0", "monthly_spend": "$180.20", "latency_ms": "95.0"},
        {"customer_id": "C107", "service_type": "Cloud_Storage", "usage_hours": "60.5", "monthly_spend": "$60.50", "latency_ms": "19.8"},
        {"customer_id": "C101", "service_type": "Cloud_Compute", "usage_hours": "120.5", "monthly_spend": "$245.80", "latency_ms": "32.4"},
        {"customer_id": "C108", "service_type": "Database", "usage_hours": "150.0", "monthly_spend": "$210.00", "latency_ms": "28.7"},
        {"customer_id": "C109", "service_type": "AI_Inference", "usage_hours": "450.0", "monthly_spend": "$920.40", "latency_ms": "140.2"},
        {"customer_id": "C110", "service_type": "Cloud_Compute", "usage_hours": "220.0", "monthly_spend": None, "latency_ms": "42.1"},
        {"customer_id": "C111", "service_type": "Cloud_Storage", "usage_hours": "15.0", "monthly_spend": "$20.00", "latency_ms": "15.5"},
        {"customer_id": "C112", "service_type": "AI_Inference", "usage_hours": "310.0", "monthly_spend": "$640.00", "latency_ms": "115.0"},
        {"customer_id": "C113", "service_type": "Database", "usage_hours": "5000.0", "monthly_spend": "$250.00", "latency_ms": "31.0"},
        {"customer_id": "C114", "service_type": "Cloud_Compute", "usage_hours": "95.0", "monthly_spend": "$190.00", "latency_ms": "35.8"},
        {"customer_id": "C115", "service_type": "AI_Inference", "usage_hours": "180.5", "monthly_spend": "$370.10", "latency_ms": "88.4"},
    ]
    return pd.DataFrame(raw_data)


def clean_currency_column(series: pd.Series) -> pd.Series:
    """Removes currency symbols ($), commas, and casts to float."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"None": np.nan, "nan": np.nan, "": np.nan})
        .astype(float)
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe using distinct data transformation functions:
    1. Removes duplicate rows based on customer_id
    2. Cleans currency format on monthly_spend
    3. Converts numeric columns to float
    4. Filters invalid negative usage and extreme outliers (> 2000 hrs)
    5. Imputes missing values with column medians
    """
    cleaned = df.copy()

    # Step 1: Remove duplicates
    cleaned = cleaned.drop_duplicates(subset=["customer_id"])

    # Step 2: Clean currency and format numeric fields
    cleaned["monthly_spend"] = clean_currency_column(cleaned["monthly_spend"])
    cleaned["usage_hours"] = pd.to_numeric(cleaned["usage_hours"], errors="coerce")
    cleaned["latency_ms"] = pd.to_numeric(cleaned["latency_ms"], errors="coerce")

    # Step 3: Handle invalid values / extreme outliers (set to NaN for imputation)
    # Usage hours cannot be negative or > 2000 for a single billing month
    cleaned.loc[cleaned["usage_hours"] <= 0, "usage_hours"] = np.nan
    cleaned.loc[cleaned["usage_hours"] > 2000, "usage_hours"] = np.nan

    # Step 4: Impute missing numerical values using median
    for col in ["usage_hours", "monthly_spend", "latency_ms"]:
        median_val = cleaned[col].median()
        cleaned[col] = cleaned[col].fillna(median_val)

    # Reset index
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def process_with_numpy(cleaned_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Performs vector and matrix numerical processing using NumPy:
    - Extracts numeric feature matrix: [usage_hours, monthly_spend, latency_ms]
    - Computes derived metric: hourly_cost_rate = monthly_spend / usage_hours (vectorized)
    - Normalizes feature matrix using Z-Score: (X - mean) / std
    - Calculates summary statistics (mean, std, min, max, percentiles)
    """
    # 1. Convert columns to NumPy 2D feature matrix (Shape: N x 3)
    feature_matrix = cleaned_df[["usage_hours", "monthly_spend", "latency_ms"]].to_numpy()

    # 2. Vectorized NumPy operation: Calculate cost per hour
    usage = feature_matrix[:, 0]
    spend = feature_matrix[:, 1]
    hourly_cost_rate = np.round(spend / usage, 2)

    # 3. Vectorized Z-Score Normalization across columns
    means = np.mean(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)
    normalized_matrix = (feature_matrix - means) / stds

    # 4. Statistical analysis using NumPy
    stats = {
        "features": ["usage_hours", "monthly_spend", "latency_ms"],
        "means": np.round(means, 2),
        "stds": np.round(stds, 2),
        "mins": np.min(feature_matrix, axis=0),
        "maxs": np.max(feature_matrix, axis=0),
        "median": np.median(feature_matrix, axis=0),
        "25th_percentile": np.percentile(feature_matrix, 25, axis=0),
        "75th_percentile": np.percentile(feature_matrix, 75, axis=0),
        "avg_hourly_cost": round(float(np.mean(hourly_cost_rate)), 2),
    }

    return feature_matrix, normalized_matrix, stats


def main():
    print("=" * 60)
    print(" AWS AI/ML Track - Week 1: Data Processing Pipeline ")
    print("=" * 60)

    # 1. Load
    raw_df = load_raw_dataset("raw_dataset.csv")
    print(f"\n[1] Raw Data Sample:\n{raw_df.head(4)}\n")

    # 2. Clean
    cleaned_df = clean_dataset(raw_df)
    print(f"[2] Cleaned Data (Total {len(cleaned_df)} records):\n{cleaned_df}\n")

    # 3. Process with NumPy
    features, normalized, stats = process_with_numpy(cleaned_df)
    print("[3] NumPy Feature Matrix Shape:", features.shape)
    print("\n[4] Summary Statistics (Computed via NumPy):")
    for idx, feature in enumerate(stats["features"]):
        print(f" - {feature.upper()}:")
        print(f"     Mean: {stats['means'][idx]} | Std: {stats['stds'][idx]}")
        print(f"     Min:  {stats['mins'][idx]} | Max: {stats['maxs'][idx]}")
        print(f"     25th Percentile: {stats['25th_percentile'][idx]} | 75th: {stats['75th_percentile'][idx]}")

    print(f"\n - Average Hourly Cost Rate ($/hr): ${stats['avg_hourly_cost']}")
    print("\n[5] Normalized Feature Matrix (First 3 rows):\n", np.round(normalized[:3], 3))
    print("\n" + "=" * 60)
    print(" Processing Complete! ")
    print("=" * 60)


if __name__ == "__main__":
    main()
