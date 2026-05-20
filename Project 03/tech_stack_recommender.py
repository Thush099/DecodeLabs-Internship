import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


class DigitalMatchmaker:
    """
    DECODELABS PROJECT 3: TECH STACK RECOMMENDER

    Objective: Shifting from passive classification to active prediction.
    Methodology: Content-Based Filtering using TF-IDF and Cosine Similarity.
    """

    def __init__(self, data_path='Raw-Skills-Csv-Dataset.csv'):
        print("[SYSTEM] Booting up the Digital Matchmaker Engine...")

        self.data_path = data_path
        self.dataset = self._load_csv_data()

        # UPGRADE: Custom Regex Tokenizer.
        # Prevents the default engine from deleting punctuation in "C++", "C#", or "CI/CD".
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w\.+#/-]*\b")

        # Mapping the items into the mathematical vector space (Bridging the language barrier)
        self.item_matrix = self.vectorizer.fit_transform(self.dataset['Required Skills'])

        print(f"[SYSTEM] Successfully vectorized {len(self.dataset)} job roles.")

    def _load_csv_data(self):
        """Loads the dataset and handles missing files gracefully."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] Cannot find '{self.data_path}'. Please check the directory.")
        return pd.read_csv(self.data_path)

    def _step1_ingestion(self, user_skills):
        """
        PIPELINE STEP 1: INGESTION
        Also mitigates the 'Achilles Heel: The Cold Start Problem'.
        """
        print(">> Executing Step 1: Ingestion...")

        # Force a minimum data density to ensure accurate angular alignment
        if not user_skills or len(user_skills) < 3:
            raise ValueError("COLD START DETECTED: Please provide at least 3 skills to bootstrap the vector.")

        user_profile_string = " ".join(user_skills)
        # Transform the user profile into the shared vocabulary space
        return self.vectorizer.transform([user_profile_string])

    def _step2_scoring(self, user_vector):
        """
        PIPELINE STEP 2: SCORING
        Uses Cosine Similarity to measure the exact mathematical angle, avoiding Euclidean magnitude flaws.
        """
        print(">> Executing Step 2: Scoring (Calculating Cosine Similarity)...")
        return cosine_similarity(user_vector, self.item_matrix).flatten()

    def _step3_sorting(self, cosine_scores):
        """
        PIPELINE STEP 3: SORTING
        Organizes the scored dataset in descending order.
        """
        print(">> Executing Step 3: Sorting...")
        results_df = self.dataset.copy()
        results_df['Raw Score'] = cosine_scores

        # Convert raw decimal to a clean percentage for commercial-grade UI output
        results_df['Match Confidence'] = np.round(cosine_scores * 100, 2).astype(str) + '%'

        return results_df.sort_values(by='Raw Score', ascending=False)

    def _step4_filtering(self, sorted_results, top_n):
        """
        PIPELINE STEP 4: FILTERING
        Truncates the list to prevent choice overload (Top-N Output).
        """
        print(f">> Executing Step 4: Filtering (Truncating to Top {top_n})...")
        return sorted_results.head(top_n)

    def run_recommendation_pipeline(self, user_skills: list, top_n: int = 3):
        """Runs the strict 4-step assembly line required by the project spec."""
        print("\n" + "=" * 50)
        print(" INITIALIZING 4-STEP RANKING PIPELINE")
        print("=" * 50)

        try:
            # 1. INGESTION
            user_vector = self._step1_ingestion(user_skills)

            # 2. SCORING
            scores = self._step2_scoring(user_vector)

            # 3. SORTING
            sorted_dataset = self._step3_sorting(scores)

            # 4. FILTERING
            final_recommendations = self._step4_filtering(sorted_dataset, top_n)

            return final_recommendations[['Job Role', 'Match Confidence']]

        except ValueError as e:
            print(f"\n{e}")
            return None



# EXECUTION SCRIPT

if __name__ == "__main__":
    # Instantiate the class
    engine = DigitalMatchmaker(data_path='Raw-Skills-Csv-Dataset.csv')

    # The user inputs (acting as the onboarding survey to bypass the Cold Start)
    my_skills = ["Python", "AWS", "Docker", "Linux", "Kubernetes"]
    print(f"\nUser Inputs Received: {my_skills}")

    # Run the engine
    top_picks = engine.run_recommendation_pipeline(user_skills=my_skills, top_n=3)

    # Final Output
    if top_picks is not None:
        print("\n" + "*" * 50)
        print(" DIGITAL MATCHMAKER OUTPUT")
        print("*" * 50)
        print(top_picks.to_string(index=False))
        print("*" * 50 + "\n")