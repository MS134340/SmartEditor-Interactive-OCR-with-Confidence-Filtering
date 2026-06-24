# SmartEditor-Interactive-OCR-with-Confidence-Filtering

**SmartEditor** is a Python-based OCR tool that combines traditional OCR with **real-time user correction**. It uses **Tesseract OCR**, confidence score filtering, and a **Tkinter GUI** to extract and edit text from scanned documents and noisy image files—reducing manual post-processing.

---

## 🧠 Key Features

- 📷 Image preprocessing: grayscale + Gaussian blur
- 🔍 Confidence-based word filtering (Tesseract)
- 🖼 Real-time OCR output with GUI editing (Tkinter)
- 💾 Saves user-edited text in `.txt` format
- 🎯 Tested on scanned documents, signage, forms, and handwritten images

---

## 🚀 How It Works

1. **Upload** a scanned image or photo-based document.
2. **Preprocessing** enhances text visibility (grayscale, denoise, threshold).
3. **OCR & Filtering**:
   - Tesseract extracts words and confidence scores.
   - Only words with confidence ≥ 60 are passed to the GUI.
4. **GUI Interaction**:
   - Preview annotated image with bounding boxes.
   - Edit extracted text live.
5. **Export** final corrected text to `edited_text.txt`.

---

## 🛠 Requirements
- Python 3.10+
- Tesseract OCR (Install & update path in script)
- Libraries:
  - pytesseract
  - opencv-python
  - numpy
  - tkinter
  - Pillow

## ⚙️ Installation & Usage
1. **Install Tesseract**
    Download: https://github.com/tesseract-ocr/tesseract
    Set path in script:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```
2. **Install Python dependencies**
    ```bash
    pip install opencv-python numpy pytesseract Pillow
3. **Run the application**
    ```bash
    python SmartEditor-Interactive-OCR-with-Confidence-Filtering.py
## 📊 Experimental Results
| Model                      | Word Accuracy | Character Accuracy |
| -------------------------- | ------------- | ------------------ |
| Default Tesseract          | 64.8%         | 77.8%              |
| Convolutional Preprocess   | —             | 61.6%              |
| Super-resolution + Tess.   | 86.0%         | 89.7%              |
| EasyOCR + Post-process     | 85–93%        | —                  |
| **SmartEditor (Proposed)** | **75%**       | **73.0%**          |

SmartEditor balances accuracy with user control, significantly reducing the burden of manual proofreading.
