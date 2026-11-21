
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# STEP 6: MODEL EVALUATION
# ==========================================

def evaluate_models(models, X_train, X_test, y_test, features_df):

    print("\n========== STEP 6: MODEL EVALUATION ==========")

    # Get training labels using original row indices
    y_train = features_df.loc[X_train.index, "match"]

    # Generate predictions
    log_train_predictions = models["Logistic Regression"].predict(X_train)
    log_predictions = models["Logistic Regression"].predict(X_test)

    rf_train_predictions = models["Random Forest"].predict(X_train)
    rf_predictions = models["Random Forest"].predict(X_test)

    # ==========================================
    # EVALUATION FUNCTION
    # ==========================================

    def evaluate_model(name, y_true, y_pred):

        print(f"\n========== {name} ==========")

        print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))

        print(
            "Precision:",
            round(precision_score(y_true, y_pred, zero_division=0), 4)
        )

        print(
            "Recall:",
            round(recall_score(y_true, y_pred, zero_division=0), 4)
        )

        print(
            "F1 Score:",
            round(f1_score(y_true, y_pred, zero_division=0), 4)
        )

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            zero_division=0
        ))

    # ==========================================
    # TEST SET EVALUATION
    # ==========================================

    evaluate_model(
        "LOGISTIC REGRESSION - TEST DATA",
        y_test,
        log_predictions
    )

    evaluate_model(
        "RANDOM FOREST - TEST DATA",
        y_test,
        rf_predictions
    )

    # ==========================================
    # TRAIN VS TEST ACCURACY
    # ==========================================

    print("\n========== TRAIN VS TEST COMPARISON ==========")

    print(
        "Logistic Regression Training Accuracy:",
        round(accuracy_score(y_train, log_train_predictions), 4)
    )

    print(
        "Logistic Regression Testing Accuracy:",
        round(accuracy_score(y_test, log_predictions), 4)
    )

    print(
        "Random Forest Training Accuracy:",
        round(accuracy_score(y_train, rf_train_predictions), 4)
    )

    print(
        "Random Forest Testing Accuracy:",
        round(accuracy_score(y_test, rf_predictions), 4)
    )

    # ==========================================
    # STEP 7: RELIABILITY CHECK
    # ==========================================

    print("\n========== STEP 7: RELIABILITY CHECK ==========")

    train_df = features_df.loc[X_train.index].copy()
    test_df = features_df.loc[X_test.index].copy()

    # Check Lost ID overlap
    common_lost_ids = (
        set(train_df["lost_id"])
        & set(test_df["lost_id"])
    )

    print("Common Lost IDs:", len(common_lost_ids))

    # Check Found ID overlap
    common_found_ids = (
        set(train_df["found_id"])
        & set(test_df["found_id"])
    )

    print("Common Found IDs:", len(common_found_ids))

    # Check exact feature patterns
    train_patterns = set(map(tuple, X_train.to_numpy()))
    test_patterns = set(map(tuple, X_test.to_numpy()))

    common_patterns = train_patterns.intersection(test_patterns)

    print("Common Feature Patterns:", len(common_patterns))

    # Check feature + label combinations
    train_records = set(
        map(
            tuple,
            pd.concat(
                [X_train, y_train],
                axis=1
            ).to_numpy()
        )
    )

    test_records = set(
        map(
            tuple,
            pd.concat(
                [X_test, y_test],
                axis=1
            ).to_numpy()
        )
    )

    common_records = train_records.intersection(test_records)

    print("Common Feature + Label Records:", len(common_records))

    # Check class distribution
    print("\nTraining Label Distribution:")
    print(y_train.value_counts())

    print("\nTesting Label Distribution:")
    print(y_test.value_counts())

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    print("\n========== FEATURE IMPORTANCE ==========")

    rf_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": models["Random Forest"].feature_importances_
    })

    rf_importance = rf_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(rf_importance.to_string(index=False))

    # ==========================================
    # STEP 8: MODEL COMPARISON
    # ==========================================

    print("\n========== STEP 8: MODEL COMPARISON ==========")

    log_results = {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, log_predictions),
        "Precision": precision_score(
            y_test, log_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, log_predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, log_predictions, zero_division=0
        )
    }

    rf_results = {
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_predictions),
        "Precision": precision_score(
            y_test, rf_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, rf_predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, rf_predictions, zero_division=0
        )
    }

    comparison_df = pd.DataFrame([
        log_results,
        rf_results
    ])

    print("\nMODEL PERFORMANCE COMPARISON:")
    print(comparison_df.to_string(index=False))

    # Select best model based on F1 Score
    best_model_name = comparison_df.loc[
        comparison_df["F1 Score"].idxmax(),
        "Model"
    ]

    print("\nBest Model Based on F1 Score:", best_model_name)

    print("\nModel evaluation and reliability check completed!")

    return comparison_df, best_model_name
