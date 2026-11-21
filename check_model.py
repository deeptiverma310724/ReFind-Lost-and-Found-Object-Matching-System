
import joblib

print("========== MODEL CHECKING ==========")

model_data = joblib.load("models/best_model.pkl")

print("\nMODEL LOADED SUCCESSFULLY!")

print("\nRequired Features:")

for feature in model_data["feature_names"]:
    print("-", feature)

print("\nModel Type:")
print(type(model_data["model"]).__name__)

print("\nModel checking completed!")
