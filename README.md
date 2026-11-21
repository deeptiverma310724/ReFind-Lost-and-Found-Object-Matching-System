## ReFind – Lost and Found Object Matching System

## Introduction

ReFind is a Python-based project developed to make the process of finding lost items easier. Sometimes people lose their belongings, but it becomes difficult to identify whether an item reported as found belongs to them.

This project tries to solve this problem by comparing the details and images of lost and found items. It shows possible matches along with their similarity scores.

## Objectives

- To make the lost and found process easier.
- To compare lost and found items using their images and text details.
- To identify items that have similar names, descriptions and categories.
- To display possible matches through a simple web interface.

## Features

- Users can submit details of lost or found items.
- Users can upload item images.
- The system compares lost items with found items and vice versa.
- Text similarity is calculated using TF-IDF and Cosine Similarity.
- Image similarity is calculated using Perceptual Hashing (pHash).
- Item categories and other available details are also compared.
- Matching results and similarity percentages are displayed.
- The application is built using Streamlit.

## Technologies Used

- **Programming Language:** Python
- **Web Interface:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Image Processing:** Pillow, ImageHash
- **Model Saving:** Joblib

## How the System Works

1. The user selects whether the item is lost or found.
2. The user enters the item details and uploads an image.
3. The system checks previously submitted items of the opposite type.
4. It calculates text and image similarity.
5. It also checks the item category and other available details.
6. The system ranks the possible matches and displays the results.

## Project Structure

- `app/` – Contains the Streamlit application.
- `src/` – Contains the Python files for preprocessing, matching, image comparison and model prediction.
- `data/` – Contains the project datasets.
- `models/` – Contains the saved machine learning model.
- `main.py` – Main project execution file.
- `requirements.txt` – Required Python libraries.

## How to Run the Project

**Step 1:** Download or clone the repository.

**Step 2:** Install the required libraries.

```bash
pip install -r requirements.txt
```

**Step 3:** Run the Streamlit application.

```bash
python -m streamlit run app/app.py
```

## Limitations

- Matching results depend on the quality of uploaded images and item descriptions.
- Very different images of the same physical object may not always be identified correctly.
- The system currently works with the item details and categories provided by users.

## Future Scope

- Add user login and registration.
- Improve image matching for different angles and lighting conditions.
- Add notifications when a possible match is found.
- Make the system available to more users through online deployment.

## Conclusion

ReFind is an attempt to use machine learning and similarity analysis to make finding lost belongings easier. The project combines text comparison, image comparison and item details to help users identify possible matches through a simple web application.

**Project Type:** Academic Project – B.Tech CSE (AI & ML)