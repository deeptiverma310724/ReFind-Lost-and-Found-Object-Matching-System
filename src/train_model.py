
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd


# ==========================================
# STEP 3: FEATURE SELECTION
# ==========================================

def feature_selection(features_df):

    print("\n========== STEP 3: FEATURE SELECTION ==========")

    required_columns = [
        "lost_id",
        "found_id",
        "match",
        "image_similarity",
        "text_similarity",
        "category_match",
        "color_match",
        "brand_match",
        "location_match",
        "time_difference_hours"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in features_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    X = features_df.drop(
        columns=["lost_id", "found_id", "match"]
    ).copy()

    y = features_df["match"].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        raise ValueError(
            "Missing or non-numeric feature values found."
        )

    if y.isnull().any():
        raise ValueError("Missing target values found.")

    print("\nInput Features:")
    print(X.columns.tolist())

    print("\nMatch Distribution:")
    print(y.value_counts())

    print("\nInput Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nFeature Correlation with Match:")
    print(
        features_df[X.columns.tolist() + ["match"]]
        .corr(numeric_only=True)["match"]
        .sort_values(ascending=False)
    )

    print("\nFeature selection completed!")

    return X, y


# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================

def split_data(X, y, features_df):

    print("\n========== STEP 4: TRAIN-TEST SPLIT ==========")

    groups = features_df["lost_id"]

    splitter = GroupShuffleSplit(
        n_splits=20,
        test_size=0.20,
        random_state=42
    )

    for train_idx, test_idx in splitter.split(X, y, groups=groups):

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        if y_train.nunique() == 2 and y_test.nunique() == 2:
            break
    else:
        raise ValueError(
            "Could not create a split containing both classes."
        )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    train_df = features_df.iloc[train_idx]
    test_df = features_df.iloc[test_idx]

    common_lost = (
        set(train_df["lost_id"])
        & set(test_df["lost_id"])
    )

    common_found = (
        set(train_df["found_id"])
        & set(test_df["found_id"])
    )

    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    print("Common Lost IDs:", len(common_lost))
    print("Common Found IDs:", len(common_found))

    if common_lost:
        raise ValueError("Lost ID leakage detected!")

    print("\nTrain-test split completed!")

    return X_train, X_test, y_train, y_test

# ==========================================
# STEP 5: MODEL TRAINING
# ==========================================

def train_models(X_train, y_train):

    print("\n========== STEP 5: MODEL TRAINING ==========")

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain both match classes."
        )

    # Logistic Regression
    log_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    log_model.fit(X_train, y_train)

    print("\nLogistic Regression training completed!")

    rf_model.fit(X_train, y_train)

    print("Random Forest training completed!")

    models = {
        "Logistic Regression": log_model,
        "Random Forest": rf_model
    }

    print("\nBoth models trained successfully!")

    return models
