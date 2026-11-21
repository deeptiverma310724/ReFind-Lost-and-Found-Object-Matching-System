
import streamlit as st
import pandas as pd
import re
import uuid

from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.image_matching import calculate_image_similarity
from src.predict import predict_from_features


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Lost and Found Matching System",
    page_icon="🔍",
    layout="centered"
)

st.title("Lost and Found Object Matching System")
st.write("Find matching lost and found items using intelligent matching!")


# ==================================================
# 2. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

data_folder = ROOT_DIR / "data" / "processed"
image_folder = data_folder / "images"
csv_path = data_folder / "items.csv"

image_folder.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "item_type",
    "item_name",
    "description",
    "category",
    "color",
    "brand",
    "location",
    "image_path",
    "submitted_at"
]


# ==================================================
# 3. STOP WORDS
# ==================================================

STOP_WORDS = {
    "the", "is", "are", "was", "were",
    "a", "an", "and", "or", "of", "to",
    "for", "from", "with", "this", "that",
    "it", "my", "our", "your", "has",
    "have", "had", "been", "being",
    "hai", "hain", "tha", "thi",
    "ka", "ki", "ke", "ko", "se",
    "mein", "par", "aur", "ek"
}

LOCATION_STOP_WORDS = {
    "near", "nearby", "at", "in", "the",
    "outside", "inside", "opposite",
    "beside", "behind", "of", "next", "to"
}


# ==================================================
# 4. LOAD SAVED ITEMS
# ==================================================

def load_items():

    if csv_path.exists():

        df = pd.read_csv(csv_path)

        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df[COLUMNS].fillna("")

    return pd.DataFrame(columns=COLUMNS)


# ==================================================
# 5. TEXT CLEANING
# ==================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).casefold().strip()

    words = re.findall(
        r"\w+",
        text,
        flags=re.UNICODE
    )

    words = [
        word for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


# ==================================================
# 6. IMAGE PATH HANDLING
# ==================================================

def get_image_path(path):

    if pd.isna(path) or not str(path).strip():
        return None

    path = Path(str(path))

    if not path.is_absolute():
        path = ROOT_DIR / path

    if path.exists() and path.is_file():
        return path

    return None


# ==================================================
# 7. CATEGORY COMPARISON
# ==================================================

def compare_categories(category1, category2):

    category1 = str(category1).strip().casefold()
    category2 = str(category2).strip().casefold()

    if not category1 or not category2:
        return 0

    # Other does not automatically match every category
    return int(category1 == category2)


# ==================================================
# 8. DETAIL COMPARISON
# ==================================================

def compare_details(value1, value2):

    value1 = clean_text(value1)
    value2 = clean_text(value2)

    if not value1 or not value2:
        return 0

    return int(value1 == value2)


# ==================================================
# 9. LOCATION COMPARISON
# ==================================================

def compare_locations(location1, location2):

    if pd.isna(location1) or pd.isna(location2):
        return 0

    words1 = set(
        re.findall(r"\w+", str(location1).casefold())
    ) - LOCATION_STOP_WORDS

    words2 = set(
        re.findall(r"\w+", str(location2).casefold())
    ) - LOCATION_STOP_WORDS

    if not words1 or not words2:
        return 0

    common = words1.intersection(words2)
    all_words = words1.union(words2)

    similarity = len(common) / len(all_words)

    return int(similarity >= 0.5)


# ==================================================
# 10. TEXT SIMILARITY
# ==================================================

def calculate_text_similarity(text1, text2):

    text1 = clean_text(text1)
    text2 = clean_text(text2)

    if not text1 or not text2:
        return 0.0

    try:

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=True
        )

        matrix = vectorizer.fit_transform(
            [text1, text2]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(score * 100, 2)

    except ValueError:

        return 0.0


# ==================================================
# 11. FIND MATCHES
# ==================================================

def find_matches(new_item, candidates):

    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.copy().reset_index(drop=True)
    results = candidates.copy()

    name_scores = []
    description_scores = []
    text_scores = []
    image_scores = []

    category_matches = []
    color_matches = []
    brand_matches = []
    location_matches = []

    time_differences = []
    ranking_scores = []

    ml_probabilities = []
    ml_predictions = []

    new_name = str(new_item["item_name"])
    new_description = str(new_item["description"])

    new_full_text = new_name + " " + new_description

    new_image = get_image_path(
        new_item["image_path"]
    )

    # ----------------------------------------------
    # COMPARE CANDIDATE ITEMS
    # ----------------------------------------------

    for _, candidate in candidates.iterrows():

        old_name = str(candidate["item_name"])
        old_description = str(candidate["description"])

        old_full_text = old_name + " " + old_description

        old_image = get_image_path(
            candidate["image_path"]
        )

        # NAME SIMILARITY

        name_score = calculate_text_similarity(
            new_name,
            old_name
        )

        name_scores.append(name_score)

        # DESCRIPTION SIMILARITY

        description_score = calculate_text_similarity(
            new_description,
            old_description
        )

        description_scores.append(description_score)

        # FULL TEXT SIMILARITY

        text_score = calculate_text_similarity(
            new_full_text,
            old_full_text
        )

        text_scores.append(text_score)

        # IMAGE SIMILARITY

        image_score = 0.0

        both_images_available = (
            new_image is not None
            and old_image is not None
        )

        if both_images_available:

            try:

                image_score = calculate_image_similarity(
                    str(new_image),
                    str(old_image)
                )

                image_score = max(
                    0.0,
                    min(100.0, float(image_score))
                )

            except Exception:

                image_score = 0.0

        image_scores.append(image_score)

        # CATEGORY

        category_match = compare_categories(
            new_item["category"],
            candidate["category"]
        )

        # COLOR / BRAND

        color_match = compare_details(
            new_item["color"],
            candidate["color"]
        )

        brand_match = compare_details(
            new_item["brand"],
            candidate["brand"]
        )

        # LOCATION

        location_match = compare_locations(
            new_item["location"],
            candidate["location"]
        )

        category_matches.append(category_match)
        color_matches.append(color_match)
        brand_matches.append(brand_match)
        location_matches.append(location_match)

        # TIME DIFFERENCE

        try:

            new_time = pd.to_datetime(
                new_item["submitted_at"]
            )

            old_time = pd.to_datetime(
                candidate["submitted_at"]
            )

            time_diff = abs(
                (new_time - old_time).total_seconds() / 3600
            )

        except Exception:

            time_diff = 0.0

        time_differences.append(time_diff)

        # RANKING SCORE

        if both_images_available:

            ranking_score = (
                0.35 * name_score
                + 0.25 * text_score
                + 0.40 * image_score
            )

        else:

            ranking_score = (
                0.40 * name_score
                + 0.60 * text_score
            )

        ranking_scores.append(
            round(ranking_score, 2)
        )

        # ML PREDICTION - SUPPORTING INFORMATION

        features = {
            "image_similarity": image_score / 100,
            "text_similarity": text_score / 100,
            "category_match": category_match,
            "color_match": color_match,
            "brand_match": brand_match,
            "location_match": location_match,
            "time_difference_hours": time_diff
        }

        try:

            prediction, confidence = predict_from_features(
                features
            )

            ml_probabilities.append(
                round(float(confidence), 2)
            )

            ml_predictions.append(
                "MATCH" if int(prediction) == 1 else "NO MATCH"
            )

        except Exception:

            ml_probabilities.append(0.0)
            ml_predictions.append("Unavailable")

    # ==================================================
    # 12. ADD RESULT COLUMNS
    # ==================================================

    results["name_similarity"] = name_scores
    results["description_similarity"] = description_scores
    results["text_similarity"] = text_scores
    results["image_similarity"] = image_scores

    results["category_match"] = category_matches
    results["color_match"] = color_matches
    results["brand_match"] = brand_matches
    results["location_match"] = location_matches

    results["time_difference_hours"] = time_differences
    results["ranking_score"] = ranking_scores

    results["ml_probability"] = ml_probabilities
    results["ml_prediction"] = ml_predictions

    # ==================================================
    # 13. MATCHING FILTER
    # ==================================================

    new_image_available = new_image is not None

    old_images_available = results["image_path"].apply(
        lambda path: get_image_path(path) is not None
    )

    both_images_available = (
        new_image_available & old_images_available
    )

    # BOTH IMAGES AVAILABLE

    image_matches = (
        both_images_available
        & (results["image_similarity"] >= 80)
        & (
            (results["name_similarity"] >= 30)
            | (results["text_similarity"] >= 35)
        )
        & (results["category_match"] == 1)
    )

    # IMAGES MISSING

    text_matches = (
        ~both_images_available
        & (results["name_similarity"] >= 60)
        & (results["text_similarity"] >= 60)
        & (results["category_match"] == 1)
    )

    results = results[
        image_matches | text_matches
    ].copy()

    results = results.sort_values(
        by="ranking_score",
        ascending=False
    )

    return results.head(5)


# ==================================================
# 14. USER INPUT FORM
# ==================================================

st.header("Report a Lost or Found Item")

with st.form("item_form"):

    item_type = st.selectbox(
        "Select Item Type",
        ["Lost Item", "Found Item"]
    )

    item_name = st.text_input(
        "Enter Item Name",
        placeholder="Example: Black Backpack"
    )

    description = st.text_area(
        "Enter Item Description",
        placeholder="Describe your item..."
    )

    st.subheader("Additional Item Details")

    category = st.selectbox(
        "Select Category",
        [
            "Bottle",
            "Book",
            "Electronics",
            "Wallet",
            "Bag",
            "Clothing",
            "Accessories",
            "Other"
        ]
    )

    color = st.text_input("Enter Item Color")

    brand = st.text_input("Enter Brand (Optional)")

    location = st.text_input("Enter Lost/Found Location")

    uploaded_file = st.file_uploader(
        "Upload Item Image",
        type=["jpg", "jpeg", "png"]
    )

    submit_button = st.form_submit_button("Submit")


# ==================================================
# 15. PROCESS SUBMISSION
# ==================================================

if submit_button:

    if not item_name.strip():

        st.error("Please enter the item name.")

    elif not description.strip():

        st.error("Please enter the item description.")

    else:

        # SAVE IMAGE

        saved_image_path = ""

        if uploaded_file is not None:

            extension = Path(
                uploaded_file.name
            ).suffix.lower()

            image_name = (
                f"{uuid.uuid4().hex}{extension}"
            )

            image_path = image_folder / image_name

            with open(image_path, "wb") as file:

                file.write(uploaded_file.getbuffer())

            saved_image_path = str(image_path)

        # CREATE NEW ITEM

        new_item = {
            "item_type": item_type,
            "item_name": item_name.strip(),
            "description": description.strip(),
            "category": category,
            "color": color.strip(),
            "brand": brand.strip(),
            "location": location.strip(),
            "image_path": saved_image_path,
            "submitted_at": datetime.now().isoformat()
        }

        # LOAD OLD RECORDS

        all_items = load_items()

        # SELECT ONLY OPPOSITE ITEM TYPE

        if item_type == "Lost Item":

            candidates = all_items[
                all_items["item_type"] == "Found Item"
            ].copy()

        else:

            candidates = all_items[
                all_items["item_type"] == "Lost Item"
            ].copy()

        # FIND MATCHES BEFORE SAVING NEW ITEM

        if not candidates.empty:

            matches = find_matches(
                new_item,
                candidates
            )

        else:

            matches = pd.DataFrame()

        # SAVE NEW ITEM

        updated_items = pd.concat(
            [
                all_items,
                pd.DataFrame([new_item])
            ],
            ignore_index=True
        )

        updated_items.to_csv(
            csv_path,
            index=False
        )

        st.success("Item submitted successfully!")

        # ==================================================
        # 16. DISPLAY MATCHING RESULTS
        # ==================================================

        st.header("Matching Results")

        if candidates.empty:

            st.info(
                "No opposite-type items are available yet. "
                "Your item has been saved."
            )

            st.markdown(
                "### **NO MATCH FOUND — 0.00%**"
            )

        elif matches.empty:

            st.warning(
                "No sufficiently similar matching items found. "
                "Your item has been saved."
            )

            st.markdown(
                "### **NO MATCH FOUND — 0.00%**"
            )

        else:

            for _, match in matches.iterrows():

                st.divider()

                st.subheader(match["item_name"])

                st.write(
                    "**Item Type:**",
                    match["item_type"]
                )

                st.write(
                    "**Name Similarity:**",
                    f'{match["name_similarity"]:.2f}%'
                )

                st.write(
                    "**Description Similarity:**",
                    f'{match["description_similarity"]:.2f}%'
                )

                st.write(
                    "**Text Similarity:**",
                    f'{match["text_similarity"]:.2f}%'
                )

                st.write(
                    "**Image Similarity:**",
                    f'{match["image_similarity"]:.2f}%'
                )

                st.write(
                    "**Category Match:**",
                    "Yes" if match["category_match"] == 1 else "No"
                )

                st.write(
                    "**Color Match:**",
                    "Yes" if match["color_match"] == 1 else "No"
                )

                st.write(
                    "**Brand Match:**",
                    "Yes" if match["brand_match"] == 1 else "No"
                )

                st.write(
                    "**Location Match:**",
                    "Yes" if match["location_match"] == 1 else "No"
                )

                st.write(
                    "**Description:**",
                    match["description"]
                )

                old_image = get_image_path(
                    match["image_path"]
                )

                if old_image:

                    st.image(
                        str(old_image),
                        width=200
                    )

                st.write(
                    f'**ML Prediction:** {match["ml_prediction"]}'
                )

                st.write(
                    f'**ML Confidence:** '
                    f'{match["ml_probability"]:.2f}%'
                )

                st.divider()

                st.markdown(
                    f'### **MATCH FOUND — '
                    f'{match["ranking_score"]:.2f}%**'
                )

        # ==================================================
        # 17. DISPLAY UPLOADED IMAGE
        # ==================================================

        if saved_image_path:

            st.divider()

            st.subheader("Uploaded Item")

            st.image(
                saved_image_path,
                width=200
            )
