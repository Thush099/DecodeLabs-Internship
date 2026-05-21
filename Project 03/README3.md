# AI Recommendation Logic – Tech Stack Recommender 

## Overview
This project is a sophisticated, content-based recommendation system engineered to map a user's raw skills to specific technology career paths and job roles. Moving beyond passive classification, this engine acts as a "Digital Matchmaker," utilizing natural language processing and mathematical angular alignment to output objective, ranked career recommendations. 

This project was built as the capstone for **Project 3: AI Recommendation Logic**.

## Features
* **Active Prediction Engine:** Maps qualitative user input to intrinsic job role attributes using pure content-based filtering.
* **Advanced Feature Extraction:** Utilizes Term Frequency-Inverse Document Frequency (TF-IDF) to mathematically penalize generic terms (e.g., "Software") and heavily reward highly specific niche skills (e.g., "PyTorch", "Kubernetes").
* **Custom Tech Tokenization:** Implements a custom Regex tokenizer (`(?u)\b\w[\w\.+#/-]*\b`) to ensure complex tech terminologies like `C++`, `C#`, `Node.js`, and `CI/CD` are not erased during text processing.
* **Magnitude-Invariant Scoring:** Uses Cosine Similarity to measure the exact angular alignment between the user's vector and the job role vectors, preventing data magnitude from skewing results.
* **Cold Start Mitigation:** Programmatically traps empty or insufficient user profiles (requiring a minimum of 3 inputs) to bootstrap a baseline vector.

## Technologies Used
* **Python 3.x**
* **Pandas** (Data manipulation and ingestion)
* **NumPy** (Numerical operations and confidence score rounding)
* **Scikit-learn** (`TfidfVectorizer`, `cosine_similarity`)

## How It Works
The architecture strictly follows a 4-Step Ranking Pipeline:
1. **Ingestion:** Captures explicit user skills, bypasses the cold start problem, and maps the profile into a shared multi-dimensional vector space.
2. **Scoring:** Calculates the Cosine Similarity between the user's vectorized profile and the dataset of job roles.
3. **Sorting:** Organizes the dataset in descending order based on the calculated similarity scores.
4. **Filtering:** Truncates the output to prevent choice overload, delivering a clean "Top-N" list of recommendations.


