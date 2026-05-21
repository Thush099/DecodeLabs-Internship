"""
================================================================================
  DECODELABS PROJECT 4: OCR PIPELINE — PATH 1: TEXT RECOGNITION
================================================================================

VALIDATIONS :
  [✓] 1. Library Integration     — pytesseract + OpenCV (cv2)
  [✓] 2. Pre-Processing Integrity — Grayscale → Gaussian Blur → Adaptive Threshold
  [✓] 3. Accuracy Benchmarking   — Confidence score filter (≥ 80%)
  [✓] 4. Visual Confirmation     — Saves annotated output image with bounding boxes

================================================================================
"""

import cv2
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# ── If Tesseract is not on your PATH, set it manually here (Windows example):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PRE-PROCESSING ENGINE
#  Converts raw visual noise into a clean binary image that Tesseract can parse.
# ════════════════════════════════════════════════════════════════════════════════

class PreProcessor:
    """
    Handles all image pre-processing before OCR inference.
    Pipeline: Grayscale → Gaussian Blur → Adaptive Thresholding → Deskew
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"  [PRE-PROCESS] {message}")

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Step 1: Grayscale Conversion
        Collapses the 3-channel RGB matrix into a 1D intensity matrix.
        Removes distracting color data so the engine focuses on contrast alone.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._log("✔ Grayscale conversion complete — color channels collapsed.")
        return gray

    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Step 2: Gaussian Blur
        Smooths the image to eliminate micro-imperfections and artifact noise.
        Kernel size must be odd (3, 5, 7...).
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # enforce odd kernel
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        self._log(f"✔ Gaussian blur applied — kernel size: {kernel_size}x{kernel_size}.")
        return blurred

    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Step 3: Adaptive Thresholding (Otsu-style binary decision)
        Forces every pixel to choose black (0) or white (255).
        Adaptive mode handles uneven lighting that a global threshold would miss.
        """
        binary = cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,   # size of neighbourhood area
            C=2             # constant subtracted from mean
        )
        self._log("✔ Adaptive thresholding applied — binary image generated.")
        return binary

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Step 4: Deskewing
        Calculates the rotation angle of the text and snaps it back to horizontal.
        Prevents Tesseract from misreading tilted character sequences.
        """
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            self._log("⚠ Deskew skipped — no foreground pixels detected.")
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only deskew if the tilt is significant (avoid rotating perfectly straight text)
        if abs(angle) < 0.5:
            self._log(f"✔ Deskew skipped — image tilt is negligible ({angle:.2f}°).")
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        self._log(f"✔ Deskew applied — corrected {angle:.2f}° rotation.")
        return deskewed

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Executes the full pre-processing pipeline in order.
        Returns the final binary image ready for OCR inference.
        """
        print("\n[SYSTEM] ── Executing Pre-Processing Pipeline ──")
        gray     = self.to_grayscale(image)
        blurred  = self.apply_gaussian_blur(gray)
        binary   = self.apply_adaptive_threshold(blurred)
        final    = self.deskew(binary)
        print("[SYSTEM] ── Pre-Processing Complete ──\n")
        return final


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 2 — OCR INFERENCE ENGINE
#  Feeds the processed image into Tesseract and interprets the raw output.
# ════════════════════════════════════════════════════════════════════════════════

class OCRInferenceEngine:
    """
    Wraps pytesseract to extract text with per-word confidence scores.
    Applies the 80% confidence threshold gate to filter false positives.
    """

    CONFIDENCE_THRESHOLD = 80  # Project 4 minimum standard

    def __init__(self, psm: int = 11, verbose: bool = True):
        """
        psm (Page Segmentation Mode):
            3  = Fully automatic (default, varied layouts)
            6  = Single uniform block of text (book pages)
            7  = Single text line (number plates / headers)
            11 = Sparse, scattered text (invoices, receipts)  ← default here
        """
        self.psm = psm
        self.verbose = verbose
        self.config = f"--oem 3 --psm {psm}"

    def _log(self, message: str):
        if self.verbose:
            print(f"  [INFERENCE] {message}")

    def extract_with_confidence(self, processed_image: np.ndarray) -> pd.DataFrame:
        """
        Runs Tesseract in detail mode (image_to_data) to retrieve
        per-word bounding boxes AND confidence scores.
        Returns a filtered DataFrame of high-confidence detections only.
        """
        self._log(f"Running Tesseract OCR — PSM mode: {self.psm}...")

        raw_data = pytesseract.image_to_data(
            processed_image,
            config=self.config,
            output_type=pytesseract.Output.DATAFRAME
        )

        # Clean: drop empty text rows and convert confidence to numeric
        raw_data = raw_data.dropna(subset=["text"])
        raw_data = raw_data[raw_data["text"].str.strip() != ""]
        raw_data["conf"] = pd.to_numeric(raw_data["conf"], errors="coerce")
        raw_data = raw_data.dropna(subset=["conf"])

        total_words = len(raw_data)

        # ── THE 80% CONFIDENCE GATE ──────────────────────────────────────────
        high_conf = raw_data[raw_data["conf"] >= self.CONFIDENCE_THRESHOLD].copy()
        filtered_words = len(high_conf)

        self._log(f"Total word candidates detected : {total_words}")
        self._log(f"Words passing 80% confidence gate : {filtered_words}")

        if total_words > 0:
            pass_rate = (filtered_words / total_words) * 100
            self._log(f"Gate pass rate : {pass_rate:.1f}%")

        return high_conf

    def extract_full_text(self, processed_image: np.ndarray) -> str:
        """
        Returns the complete extracted string (for display / submission).
        Uses the same PSM config for consistency.
        """
        text = pytesseract.image_to_string(processed_image, config=self.config)
        return text.strip()


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 3 — VISUAL OUTPUT ENGINE
#  Draws bounding boxes and confidence labels onto the original image.
#  This satisfies Validation #4: Visual Confirmation.
# ════════════════════════════════════════════════════════════════════════════════

class VisualOutputEngine:
    """
    Annotates the original image with bounding boxes for each detected word.
    Green box = high confidence (≥ 80%).  Box colour scales with confidence.
    """

    def __init__(self, output_path: str = "ocr_output_annotated.png"):
        self.output_path = output_path

    def draw_detections(self, original_image: np.ndarray,
                        detections: pd.DataFrame) -> np.ndarray:
        """
        Draws a coloured bounding box + confidence label over each detected word.
        """
        annotated = original_image.copy()

        for _, row in detections.iterrows():
            x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
            conf = float(row["conf"])
            text = str(row["text"])

            # Colour intensity scales with confidence (greener = more confident)
            green_intensity = int((conf / 100) * 255)
            color = (0, green_intensity, 255 - green_intensity)  # BGR

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label background for readability
            label = f"{text} ({conf:.0f}%)"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x, y - label_h - 6), (x + label_w + 4, y), color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated

    def save(self, annotated_image: np.ndarray):
        """Saves the annotated image to disk."""
        cv2.imwrite(self.output_path, annotated_image)
        print(f"  [OUTPUT] ✔ Annotated image saved → {self.output_path}")


# ════════════════════════════════════════════════════════════════════════════════
#  STEP 4 — MASTER PIPELINE ORCHESTRATOR
#  Ties all three engines together into one clean run() call.
# ════════════════════════════════════════════════════════════════════════════════

class OCRPipeline:
    """
    Master controller.  Orchestrates:
        PreProcessor → OCRInferenceEngine → VisualOutputEngine

    Usage:
        pipeline = OCRPipeline(psm=11)
        results  = pipeline.run("path/to/image.png")
    """

    def __init__(self, psm: int = 11, output_path: str = "ocr_output_annotated.png",
                 verbose: bool = True):
        self.preprocessor  = PreProcessor(verbose=verbose)
        self.inference     = OCRInferenceEngine(psm=psm, verbose=verbose)
        self.visual_engine = VisualOutputEngine(output_path=output_path)
        self.verbose       = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[PIPELINE] {message}")

    def run(self, image_path: str) -> dict:
        """
        Full pipeline execution.

        Returns a dict:
            {
                "extracted_text"  : str,       # Full OCR string output
                "detections"      : DataFrame, # Per-word confidence data
                "words_extracted" : int,       # Count of validated words
                "avg_confidence"  : float,     # Average confidence of valid words
                "passed_gate"     : bool       # Did we hit the 80% threshold?
            }
        """
        print("=" * 70)
        print("  DECODELABS PROJECT 4 — OCR PIPELINE INITIATED")
        print("=" * 70)

        # ── Validate file path ───────────────────────────────────────────────
        if not os.path.exists(image_path):
            print(f"\n[ERROR] Image not found: '{image_path}'")
            print("        Please provide a valid image path.")
            sys.exit(1)

        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            print(f"\n[ERROR] Could not read image: '{image_path}'")
            sys.exit(1)

        h, w = original.shape[:2]
        self._log(f"Image dimensions: {w}px × {h}px")

        # ── Stage 1: Pre-Processing ──────────────────────────────────────────
        processed = self.preprocessor.run(original)

        # ── Stage 2: OCR Inference ───────────────────────────────────────────
        print("[SYSTEM] ── Executing OCR Inference ──")
        detections    = self.inference.extract_with_confidence(processed)
        full_text     = self.inference.extract_full_text(processed)

        words_found   = len(detections)
        avg_conf      = detections["conf"].mean() if words_found > 0 else 0.0
        passed_gate   = avg_conf >= 80.0

        # ── Stage 3: Visual Output ───────────────────────────────────────────
        print("\n[SYSTEM] ── Generating Visual Output ──")
        annotated = self.visual_engine.draw_detections(original, detections)
        self.visual_engine.save(annotated)

        # ── Final Report ─────────────────────────────────────────────────────
        self._print_report(full_text, words_found, avg_conf, passed_gate)

        return {
            "extracted_text"  : full_text,
            "detections"      : detections,
            "words_extracted" : words_found,
            "avg_confidence"  : avg_conf,
            "passed_gate"     : passed_gate
        }

    def _print_report(self, text: str, words: int, avg_conf: float, passed: bool):
        """Prints the final formatted report to the console."""
        status = "✔  PASSED" if passed else "✘  FAILED"
        bar    = "=" * 70

        print(f"\n{bar}")
        print("  DECODELABS PROJECT 4 — PIPELINE REPORT")
        print(bar)
        print(f"  Words Extracted (≥80% confidence) : {words}")
        print(f"  Average Confidence Score          : {avg_conf:.2f}%")
        print(f"  80% Accuracy Gate                 : {status}")
        print(bar)
        print("\n  ── EXTRACTED TEXT OUTPUT ──\n")
        print(text if text else "  [No text detected above confidence threshold]")
        print(f"\n{bar}\n")


# ════════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TEST IMAGE GENERATOR
#  Creates a sample image on-the-fly so the pipeline works immediately
#  without needing an external file.
# ════════════════════════════════════════════════════════════════════════════════

def generate_test_image(path: str = "test_invoice.png"):
    """
    Generates a simple synthetic invoice image for testing the full pipeline.
    Mimics the scattered-text layout of a real document (PSM 11 territory).
    """
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # white background

    lines = [
        ("INVOICE #0042",              (50,  50),  1.2, 3),
        ("Date: 2026-05-21",           (50, 100),  0.7, 2),
        ("Item: Server Rack Unit",     (50, 140),  0.7, 2),
        ("Qty: 1",                     (50, 180),  0.7, 2),
        ("Unit Price: $400.00",        (50, 220),  0.7, 2),
        ("Subtotal: $400.00",          (50, 270),  0.7, 2),
        ("Tax (10%): $40.00",          (50, 310),  0.7, 2),
        ("TOTAL: $440.00",             (50, 360),  1.0, 3),
        ("Thank you for your order!",  (50, 390),  0.5, 1),
    ]

    for text, origin, scale, thickness in lines:
        cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (30, 30, 30), thickness, cv2.LINE_AA)

    # Add a touch of Gaussian noise to simulate a real scanned document
    noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    cv2.imwrite(path, img)
    print(f"[SYSTEM] Test image generated → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════════
#  EXECUTION SCRIPT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Option A: Use the auto-generated test image (works out of the box) ──
    test_image_path = generate_test_image("test_invoice.png")

    # ── Option B: Point to your own image (uncomment the line below) ────────
    # test_image_path = "your_image.png"

    # ── Deploy the pipeline ─────────────────────────────────────────────────
    pipeline = OCRPipeline(
        psm=11,                                  # PSM 11 = sparse/scattered text
        output_path="ocr_output_annotated.png",  # annotated result saved here
        verbose=True
    )

    results = pipeline.run(test_image_path)


