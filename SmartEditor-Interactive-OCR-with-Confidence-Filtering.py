"""
SmartEditor: Interactive OCR with Confidence Filtering

This script provides a GUI application that uses Tesseract OCR to extract text
from images. It features an image preprocessing pipeline (upscaling, denoising,
adaptive thresholding, and gamma correction) to enhance text recognition, and
provides a Tkinter interface for users to review and edit the extracted text.
"""
import os
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ANNOTATED_FILENAME = "annotated_document.jpg"
TEXT_FILENAME = "edited_text.txt"

UPSCALE = 2
PSM = 6
DENOISE_STRENGTH = 30
TEMPLATE_SIZE = 7
SEARCH_SIZE = 21
GAMMA = 1.5

def load_image(path):
    """
    Loads an image from the specified file path using OpenCV.

    Args:
        path (str): The file path to the image.

    Returns:
        numpy.ndarray: The loaded image array in BGR format.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    return img

def upscale(img):
    """
    Upscales the image by a defined multiplier to improve OCR accuracy.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The upscaled image using cubic interpolation.
    """
    h, w = img.shape[:2]
    return cv2.resize(img, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)

def prepare(img):
    """
    Preprocesses the image for better OCR results.
    
    The pipeline includes:
    1. Upscaling
    2. Conversion to grayscale
    3. Non-Local Means Denoising
    4. Adaptive Mean Thresholding
    5. Gamma Correction

    Args:
        img (numpy.ndarray): The original input image.

    Returns:
        numpy.ndarray: The preprocessed binary-like image optimized for OCR.
    """
    img = upscale(img)
    # Convert image to grayscale for simpler processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Fast Non-Local Means Denoising to remove background noise
    denoised = cv2.fastNlMeansDenoising(gray, None, DENOISE_STRENGTH, TEMPLATE_SIZE, SEARCH_SIZE)
    # Apply adaptive thresholding to separate text from background
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Apply gamma correction to increase contrast
    lut = np.array([((i / 255) ** (1 / GAMMA)) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(thresh, lut)

def ocr(img):
    """
    Extracts text from the image using Tesseract OCR.

    Args:
        img (numpy.ndarray): The preprocessed image.

    Returns:
        str: The extracted text.
    """
    return pytesseract.image_to_string(img, config=f"--oem 3 --psm {PSM}")

def mark_text(img, out_name):
    """
    Draws bounding boxes around detected characters and saves the annotated image.

    Args:
        img (numpy.ndarray): The preprocessed image.
        out_name (str): The file path to save the annotated image.
    """
    copy = img.copy()
    h, w = copy.shape[:2]
    for line in pytesseract.image_to_boxes(copy).splitlines():
        char, x1, y1, x2, y2 = line.split()[:5]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        y1, y2 = h - y1, h - y2
        cv2.rectangle(copy, (x1, y2), (x2, y1), (0, 0, 255), 2)
        cv2.putText(copy, char, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite(out_name, copy)

class SmartEditor:
    """
    A Tkinter-based Graphical User Interface for the SmartEditor OCR Tool.

    This class provides the application window where users can upload an image,
    process it through the OCR pipeline, preview the bounding box annotations,
    and edit/save the extracted text.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Editor Tool")
        self.root.geometry("950x750")
        self.image_path = None
        self.setup()

    def setup(self):
        """
        Initializes and configures the Tkinter GUI layout and widgets.
        """
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("TFrame", background="#f0f0f0")

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        top = ttk.Frame(main)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for i in range(3): top.columnconfigure(i, weight=1)

        ttk.Button(top, text="Upload Image", command=self.upload).grid(row=0, column=0, padx=5)
        ttk.Button(top, text="Process Image", command=self.extract).grid(row=0, column=1, padx=5)
        ttk.Button(top, text="Preview Annotated", command=self.preview).grid(row=0, column=2, padx=5)

        self.text = scrolledtext.ScrolledText(main, font=("Segoe UI", 12), wrap=tk.WORD)
        self.text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        bottom = ttk.Frame(main)
        bottom.grid(row=2, column=0, sticky="ew")
        ttk.Button(bottom, text="Save Text", command=self.save).pack(padx=5)

    def upload(self):
        """
        Opens a file dialog to allow the user to select an image file.
        Updates the internal image path state.
        """
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.image_path = path
            messagebox.showinfo("Selected", path)

    def extract(self):
        """
        Executes the OCR pipeline on the uploaded image.
        Preprocesses the image, extracts text, generates the annotated preview image,
        and populates the text area with the OCR results.
        """
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return
        img = load_image(self.image_path)
        processed = prepare(img)
        text = ocr(processed)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, text)
        mark_text(processed, ANNOTATED_FILENAME)
        messagebox.showinfo("Done", "Extraction complete.")

    def preview(self):
        """
        Opens a new top-level window to display the annotated image with
        bounding boxes around detected text.
        """
        if not os.path.exists(ANNOTATED_FILENAME):
            messagebox.showerror("Error", "Annotated image not found.")
            return
        win = tk.Toplevel(self.root)
        win.title("Preview")
        win.geometry("850x650")
        img = Image.open(ANNOTATED_FILENAME)
        img.thumbnail((800, 600))
        tk_img = ImageTk.PhotoImage(img)
        label = ttk.Label(win, image=tk_img)
        label.image = tk_img
        label.pack(padx=10, pady=10)

    def save(self):
        """
        Saves the currently displayed text in the text editor area to a text file.
        """
        content = self.text.get("1.0", tk.END).strip()
        with open(TEXT_FILENAME, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Saved", TEXT_FILENAME)

if __name__ == "__main__":
    root = tk.Tk()
    SmartEditor(root)
    root.mainloop()
