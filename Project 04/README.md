# AI Optic Nerve – Optical Character Recognition (OCR) Pipeline

## Overview
Capstone for **Artificial Intelligence Project 4: Image/Text Recognition** — an object-oriented Computer Vision pipeline designed to extract high-confidence text from unstructured, noisy document images.

Moving beyond raw API calls, the architecture separates image preprocessing, Tesseract OCR inference, confidence gating (≥ 80%), and bounding-box visualization.

## Key Features
- **Pre-Processing Pipeline:** Grayscale conversion, Gaussian blurring, adaptive thresholding, and rotational deskewing (`OpenCV`).
- **Accuracy Benchmarking Gate:** Filters per-word detections to retain only **≥ 80% confidence** extractions, dropping background artifacts.
- **Visual Verification:** Draws confidence-scaled bounding boxes and text labels onto the source document (`ocr_output_annotated.png`).
- **Built-in Test Generator:** Synthesizes a noisy sample invoice (`test_invoice.png`) so the full pipeline runs out of the box.

## Pipeline Architecture
1. **`PreProcessor`**: Grayscale → Gaussian blur → adaptive thresholding → minimum-area deskew.
2. **`OCRInferenceEngine`**: Runs Tesseract (`--oem 3 --psm 11`) via `image_to_data` and enforces `CONFIDENCE_THRESHOLD = 80`.
3. **`VisualOutputEngine`**: Renders bounding boxes and per-word confidence tags onto the original image.
4. **`OCRPipeline`**: Orchestrates all stages and prints summary extraction metrics.

## Setup & Usage

### 1. Install Tesseract OCR
- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **Windows:** Install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to `PATH`.

### 2. Run the Pipeline
```bash
git clone https://github.com/Thush099/DecodeLabs-Internship.git
cd "DecodeLabs-Internship/Project 04"
pip install opencv-python pytesseract pandas numpy pillow
python ocr_pipeline.py
```
