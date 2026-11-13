# DecodeLabs AI Engineering Portfolio

Portfolio of four applied AI engineering projects built during the **DecodeLabs Artificial Intelligence Virtual Internship** (`Student ID: A10526305`), progressing from deterministic rule systems to tabular classification, TF-IDF recommendation, and OpenCV + Tesseract document OCR.

---

## Repository Structure

```text
DecodeLabs-Internship/
├── Project 01/
│   └── Project01.py                   # Deterministic rule-based chatbot ("White Box" intent engine)
├── Project 02/
│   ├── classification_model.py        # KNN Iris classifier with StandardScaler & confusion matrix
│   └── requirements.txt
├── Project 03/
│   ├── Raw-Skills-Csv-Dataset.csv     # Multi-role tech skills dataset
│   ├── tech_stack_recommender.py      # 4-stage TF-IDF + Cosine Similarity career recommender
│   └── README3.md                     # Project 3 design notes
├── Project 04/
│   ├── ocr_pipeline.py                # OOP OpenCV preprocessing + Tesseract OCR + ≥80% confidence gate
│   ├── test_invoice.png               # Synthetic noisy invoice test input
│   ├── ocr_output_annotated.png       # Annotated bounding-box visual confirmation
│   └── README.md                      # Project 4 setup & architecture
└── README.md
```

---

## Projects Summary

| Project | Focus | Key Techniques |
|---|---|---|
| **Project 01** — Deterministic Chatbot | Control flow & guardrails | Explicit intent routing, zero-hallucination deterministic responses |
| **Project 02** — Data Classification | Supervised ML | `StandardScaler`, `KNeighborsClassifier`, stratified split, confusion matrix heatmap |
| **Project 03** — Tech Stack Recommender | Content-based filtering | Custom regex tokenizer preserving `C++` / `C#` / `CI/CD`, TF-IDF weighting, Cosine Similarity, cold-start guard |
| **Project 04** — Computer Vision OCR | Document extraction | Grayscale, Gaussian blur, adaptive thresholding, deskewing, Tesseract PSM 11, **≥80%** confidence filter |

---

## Quick Start (Projects 03 & 04)

```bash
# Project 03 — Tech Stack Recommender
cd "Project 03"
pip install pandas numpy scikit-learn
python tech_stack_recommender.py

# Project 04 — Document OCR Pipeline (requires Tesseract installed)
cd "../Project 04"
pip install opencv-python pytesseract pandas numpy pillow
python ocr_pipeline.py
```

### Project 04 Annotated OCR Output
![Annotated OCR Output](./Project%2004/ocr_output_annotated.png)
