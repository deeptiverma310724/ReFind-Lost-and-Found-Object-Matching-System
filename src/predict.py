
import os
import joblib


# ==========================================
# STEP 9: MODEL SAVING
# ==========================================

def save_best_model(models, best_model_name, X_train):

    print("\n========== STEP 9: MODEL SAVING ==========")

    # Create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # Select the best model
    best_model = models[best_model_name]

    # Save model and its feature names together
    model_data = {
        "model": best_model,
        "feature_names": list(X_train.columns)
    }

    model_path = "models/best_model.pkl"

    joblib.dump(model_data, model_path)

    print("\nSelected Model:", best_model_name)
    print("Model saved at:", model_path)
    print("\nModel saving completed!")

    return model_path


# ==========================================
# STEP 10: PREDICTION SYSTEM
# ==========================================

def predict_match():

    import joblib
    import pandas as pd

    print("\n========== STEP 10: PREDICTION SYSTEM ==========")

    # Load saved model and feature names
    model_data = joblib.load("models/best_model.pkl")

    loaded_model = model_data["model"]
    feature_names = model_data["feature_names"]

    print("\nEnter item matching features:")

    # Take feature values from user
    values = {}

    for feature in feature_names:

        if feature == "time_difference_hours":
            values[feature] = float(
                input("Time Difference in Hours: ")
            )

        elif feature in [
            "category_match",
            "color_match",
            "brand_match",
            "location_match"
        ]:
            values[feature] = int(
                input(f"{feature} (0 or 1): ")
            )

        else:
            values[feature] = float(
                input(f"{feature} (0-1): ")
            )

    # Prepare input in training feature order
    new_data = pd.DataFrame(
        [values],
        columns=feature_names
    )

    # Predict match
    prediction = loaded_model.predict(new_data)[0]

    print("\nPrediction Result:")

    if prediction == 1:
        print("MATCH FOUND!")
    else:
        print("NO MATCH FOUND!")

    print("\nPrediction completed!")

import pandas as pd
import joblib

def predict_from_features(values):

    model_data = joblib.load("models/best_model.pkl")

    model = model_data["model"]
    feature_names = model_data["feature_names"]

    input_data = pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    match_index = list(model.classes_).index(1)

    confidence = probability[match_index] * 100

    return prediction, confidence