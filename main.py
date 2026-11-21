
# ==========================================
# LOST AND FOUND OBJECT MATCHING SYSTEM
# ==========================================

# STEP 0: IMPORT LIBRARIES

import pandas as pd
import numpy as np

from src.data_preprocessing import load_datasets, check_data_quality
from src.train_model import feature_selection, split_data, train_models
from src.evaluate_model import evaluate_models
from src.predict import save_best_model, predict_match
from src.matching import rank_matches


print("=" * 55)
print("       LOST AND FOUND OBJECT MATCHING SYSTEM")
print("=" * 55)


# ==========================================
# STEP 1: DATASET LOADING
# ==========================================

pairs_df, features_df = load_datasets()


# ==========================================
# DATASET INFORMATION
# ==========================================

print("\n========== DATASET INFORMATION ==========")

print("Total Rows:", len(features_df))
print("Unique Lost IDs:", features_df["lost_id"].nunique())
print("Unique Found IDs:", features_df["found_id"].nunique())


# ==========================================
# STEP 2: DATA QUALITY CHECKING
# ==========================================

check_data_quality(pairs_df, features_df)


# ==========================================
# DATASET LABEL VERIFICATION
# ==========================================

print("\n========== LABEL VERIFICATION ==========")

# Merge both datasets using Lost ID and Found ID

merged = pairs_df.merge(
    features_df[["lost_id", "found_id", "match"]],
    on=["lost_id", "found_id"],
    how="inner",
    suffixes=("_pairs", "_features")
)

print("Matched Rows:", len(merged))

# Verify whether same_object and match labels agree

if len(merged) > 0:

    agreement = (
        merged["same_object"] == merged["match_features"]
    ).mean() * 100

    print(
        "Same Object vs Match Agreement:",
        round(agreement, 2),
        "%"
    )

else:
    print("No common records found for label verification.")

# Display label distribution

print("\nMatch Distribution:")

print(features_df["match"].value_counts())


# ==========================================
# STEP 3: FEATURE SELECTION
# ==========================================

print("\n========== FEATURE SELECTION ==========")

X, y = feature_selection(features_df)

print("Selected Features:", list(X.columns))
print("Total Features:", X.shape[1])


# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================

print("\n========== TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    features_df
)


# ==========================================
# STEP 5: MODEL TRAINING
# ==========================================

print("\n========== MODEL TRAINING ==========")

models = train_models(
    X_train,
    y_train
)


# ==========================================
# STEP 6, 7, 8: MODEL EVALUATION,
# RELIABILITY CHECK AND COMPARISON
# ==========================================

comparison_df, best_model_name = evaluate_models(
    models,
    X_train,
    X_test,
    y_test,
    features_df
)

print("\n========== BEST MODEL ==========")
print("Selected Model:", best_model_name)


# ==========================================
# STEP 9: MODEL SAVING
# ==========================================

print("\n========== MODEL SAVING ==========")

save_best_model(
    models,
    best_model_name,
    X_train
)


# ==========================================
# STEP 10: PREDICTION SYSTEM
# ==========================================

print("\n========== PREDICTION SYSTEM ==========")

predict_match()


# ==========================================
# STEP 11: MATCHING AND RANKING
# ==========================================

print("\n========== MATCHING AND RANKING ==========")

lost_id = input("\nEnter Lost Item ID: ").strip()

ranked_matches = rank_matches(
    features_df,
    lost_id,
    top_n=5
)


# ==========================================
# PROJECT EXECUTION COMPLETED
# ==========================================

print("\n" + "=" * 55)
print("PROJECT PIPELINE EXECUTION COMPLETED!")
print("=" * 55)
