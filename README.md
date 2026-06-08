# DecodeLabs AI Engineering Portfolio 🧠💻

Welcome to my Artificial Intelligence and Data Science portfolio repository. This repository compiles the core production-grade capstones developed and verified during my **DecodeLabs Artificial Intelligence Virtual Internship (Student ID: A10526305)**. 

Moving beyond basic scripts and theoretical models, these projects emphasize the practical application of data pipelines, mathematical similarity matrix scoring, and real-world computer vision architectures.

---

## 📂 Repository Architecture

To navigate the implementation files, the codebase is structured into decoupled, isolated directories:

```text
├── Project-03-Tech-Stack-Recommender/
│   ├── Raw-Skills-Csv-Dataset.csv        # 20-row multidimensional industry skills dataset
│   └── tech_stack_recommender.py         # 4-step content-based recommendation pipeline
│
├── Project-04-Computer-Vision-OCR/
│   ├── ocr_pipeline.py                   # Object-oriented document processing & OCR engine
│   └── ocr_output_annotated.png         # Bounding box visual confirmation verification image
│
└── README.md                             # Master portfolio documentation

🛠️ Global Prerequisites & Installation Strategy
To execute or test any of the intelligent sub-systems locally, configure your workspace environment using the following steps:

1. OS-Level Binary Dependencies
The Computer Vision engine relies on a localized instance of Google's Tesseract OCR binary. This must be present on your system's PATH variable before launching the Python scripts:

macOS (Homebrew): brew install tesseract

Windows: Install via the verified binaries at UB-Mannheim/tesseract and append to environment variables.

Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr

2. Virtual Environment & Python Packages
Isolate your dependency space and compile the required numerical processing, matrix manipulation, and data science libraries:

Bash
# Initialize the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install global framework packages
pip install opencv-python pytesseract pandas numpy scikit-learn pillow
🔍 Core Portfolio Summaries
💻 Project 03: Personalization Phase — Tech Stack Recommender
Domain: Natural Language Processing (NLP) & Feature Engineering

Objective: Move away from passive data sorting into active mathematical matchmaking by converting raw textual user interests into multi-dimensional numerical vector spaces.

Core Logic: Implements TF-IDF Vectorization using a custom regular expression token pattern ((?u)\b\w[\w\.+#/-]*\b) to safeguard technical terms like C++ or CI/CD. It then computes the Cosine Similarity to measure the exact angular alignment between user inputs and career definitions, mathematically penalizing common words while systematically avoiding collaborative data loops or user cold-start failure states.

👁️ Project 04: Unstructured Data Phase — Computer Vision OCR Pipeline
Domain: Computer Vision (CV) & Document Intelligence

Objective: Transform raw, noisy, or warped document images into perfectly formatted, structured data frames using a rigorous Input-Processing-Output (IPO) model.

Core Logic: Uses OpenCV (cv2) to coordinate a multi-layer image preparation pipeline consisting of Grayscale conversion, Gaussian blurring, and local-neighborhood Adaptive Thresholding to eliminate visual shadows. It utilizes a contour-bounding algorithm for rotational document deskewing before running model inference via Tesseract OCR. Extracted text strings are subjected to an uncompromising 80% minimum confidence accuracy gate to filter out low-fidelity visual noise before outputting final tabular records and annotated imagery.

🎓 Credentials & Engineering Validations
All code assets contained in this portfolio have been executed, verified, and benchmarked against the strict quality guidelines outlined by the DecodeLabs internship curriculum.

Internship Duration: May 01, 2026 – May 31, 2026

Verification ID: A10526305

Primary Tech Stack: Python 3.x, OpenCV, PyTesseract, Pandas, NumPy, Scikit-learn
