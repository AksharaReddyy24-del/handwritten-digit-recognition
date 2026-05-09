"""Graphical user interface for handwritten digit recognition.

This module creates a simple desktop application using Tkinter.  It loads a
pre‑trained CNN model (exported from `train_model.py`), allows the user to
browse for an image file containing a handwritten digit, preprocesses the
image to match the training format and displays the predicted digit on the
screen.

Run this script after training the model with `train_model.py`.  The
application expects to find the model weights in `mnist_cnn_model.pth` in
the same directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, ImageTk
import torch

# Import the CNN class from the training script.  If you modify the network
# architecture in train_model.py you should keep the same class definition
# name here so that the weights can be loaded correctly.
from train_model import CNN, MODEL_PATH


class DigitRecognizerApp:
    """Tkinter application for handwritten digit prediction."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("400x300")
        self.root.resizable(False, False)

        # Load the trained model
        self.model = CNN()
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model Missing",
                f"Trained model not found at {MODEL_PATH}.\n"
                "Please run train_model.py first to generate the weights."
            )
            raise SystemExit
        # Load weights onto CPU
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        self.model.eval()

        # UI elements
        self.upload_button = tk.Button(
            root,
            text="Upload Image",
            command=self.upload_image,
            font=("Arial", 12),
            width=15
        )
        self.upload_button.pack(pady=10)

        # Canvas to display the uploaded image
        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

        # Label to display prediction result
        self.result_label = tk.Label(
            root,
            text="Upload an image to start",
            font=("Arial", 16),
            fg="green"
        )
        self.result_label.pack(pady=10)

        # Hold reference to PhotoImage to prevent garbage collection
        self.tk_image: Optional[ImageTk.PhotoImage] = None

    def upload_image(self) -> None:
        """Handle the Upload Image button click."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if not file_path:
            return
        try:
            # Open and display the image
            pil_img = Image.open(file_path).convert("L")  # convert to grayscale
            # Resize to 28×28 for the model.  We do not invert colours here
            # because the model was trained on images where the digit is
            # represented by higher intensities (darker strokes) and the
            # background is lighter.  If your digits appear inverted, you
            # can uncomment the following line:
            # pil_img = ImageOps.invert(pil_img)
            pil_resized = pil_img.resize((28, 28), Image.LANCZOS)

            # Display a thumbnail (we use the original aspect ratio but scale down)
            # Create a copy to avoid modifying the resized version used for prediction
            display_img = pil_resized.resize((140, 140), Image.NEAREST)
            self.tk_image = ImageTk.PhotoImage(display_img)
            self.canvas.config(image=self.tk_image)

            # Preprocess for model: convert to tensor and normalise to [0, 1].
            # The PIL image has pixel values in [0, 255], where 0 is white and
            # 255 is black.  Dividing by 255 normalises the values.  We add
            # channel and batch dimensions to shape (1, 1, 28, 28).
            img_array = np.array(pil_resized).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = int(torch.argmax(probabilities, dim=1).item())

            # Update the result label
            self.result_label.config(
                text=f"Predicted Digit: {predicted_class}",
                fg="blue"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")


def main() -> None:
    """Launch the Tkinter digit recogniser GUI."""
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()