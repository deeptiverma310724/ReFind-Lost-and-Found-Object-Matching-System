
import pandas as pd


# ==========================================
# STEP 11: MATCHING AND RANKING SYSTEM
# ==========================================

def rank_matches(features_df, lost_id, top_n=5):

    print("\n========== STEP 11: MATCHING AND RANKING ==========")

    # Check required columns
    required_columns = [
        "lost_id",
        "found_id",
        "match",
        "image_similarity",
        "text_similarity",
        "category_match",
        "color_match",
        "brand_match",
        "location_match"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in features_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Find records related to selected Lost ID
    candidates = features_df[
        features_df["lost_id"].astype(str).str.strip()
        == str(lost_id).strip()
    ].copy()

    if candidates.empty:
        print("No matching records found for this Lost ID.")
        return candidates

    # Similarity feature columns
    score_columns = [
        "image_similarity",
        "text_similarity",
        "category_match",
        "color_match",
        "brand_match",
        "location_match"
    ]

    # Convert feature values to numeric
    for col in score_columns:
        candidates[col] = pd.to_numeric(
            candidates[col],
            errors="coerce"
        )

    # Calculate ranking score
    # Image and text similarity get higher importance
    weights = {
        "image_similarity": 0.30,
        "text_similarity": 0.25,
        "category_match": 0.15,
        "color_match": 0.10,
        "brand_match": 0.10,
        "location_match": 0.10
    }

    weighted_score = 0
    total_weight = 0

    for col, weight in weights.items():

        valid_values = candidates[col].notna()

        weighted_score += (
            candidates[col].fillna(0) * weight
        )

        total_weight += valid_values * weight

    candidates["ranking_score"] = (
        weighted_score / total_weight.replace(0, float("nan"))
    ) * 100

    candidates["ranking_score"] = (
        candidates["ranking_score"].fillna(0)
    )

    # Sort highest score first
    candidates = candidates.sort_values(
        by="ranking_score",
        ascending=False
    )

    # Display results
    print("\nTop Matching Found Items:")

    display_columns = [
        "lost_id",
        "found_id",
        "ranking_score",
        "match"
    ]

    print(
        candidates[display_columns]
        .head(top_n)
        .to_string(index=False)
    )

    print("\nMatching and ranking completed!")

    return candidates.head(top_n)
