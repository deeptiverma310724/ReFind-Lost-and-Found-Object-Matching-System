import pandas as pd


def load_datasets():

    print("-" * 50)
    print("LOST AND FOUND OBJECT MATCHING SYSTEM")
    print("-" * 50)

    # Load datasets
    pairs_df = pd.read_csv("data/raw/pairs_text.csv")
    features_df = pd.read_csv("data/raw/final_features_real.csv")

    # Display dataset information
    print("\nPAIR DATASET")
    print("Shape:", pairs_df.shape)
    print("\nColumns:")
    print(pairs_df.columns.tolist())

    print("\nFEATURE DATASET")
    print("Shape:", features_df.shape)
    print("\nColumns:")
    print(features_df.columns.tolist())

    # Display first five records
    print("\nFirst 5 rows of Pair Dataset:")
    print(pairs_df.head())

    print("\nFirst 5 rows of Feature Dataset:")
    print(features_df.head())

    print("\nDataset loading completed!")

    return pairs_df, features_df


# STEP 2: DATA QUALITY CHECKING

def check_data_quality(pairs_df, features_df):

    print("\n========== PAIR DATASET CHECK ==========")

    # Check missing values
    print("\nMissing Values:")
    print(pairs_df.isnull().sum())

    # Check duplicate records
    print("\nDuplicate Rows:")
    print(pairs_df.duplicated().sum())

    # Match distribution
    print("\nMatch Distribution:")
    print(pairs_df["match"].value_counts())

    # Statistical summary
    print("\nStatistical Summary:")
    print(pairs_df.describe())

    print("\n========== FEATURE DATASET CHECK ==========")

    # Check missing values
    print("\nMissing Values:")
    print(features_df.isnull().sum())

    # Check duplicate records
    print("\nDuplicate Rows:")
    print(features_df.duplicated().sum())

    # Match distribution
    print("\nMatch Distribution:")
    print(features_df["match"].value_counts())

    print("\nData quality checking completed!")