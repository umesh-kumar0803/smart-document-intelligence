"""
Image preprocessing module for document enhancement prior to OCR.
Supports both OpenCV (if available) and pure PIL / NumPy / SciPy fallbacks.
"""
import io
import math
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Check for OpenCV availability
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class ImageProcessor:
    """Preprocesses images to maximize OCR character recognition accuracy."""

    def __init__(self):
        self.use_opencv = OPENCV_AVAILABLE

    def load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Converts raw bytes into a PIL Image."""
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def preprocess_image(
        self,
        image: Image.Image,
        deskew: bool = True,
        binarize: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True
    ) -> Tuple[Image.Image, dict]:
        """
        Executes full preprocessing pipeline on a document image.
        
        Returns:
            Tuple[Image.Image, dict]: (Preprocessed Image, Preprocessing metadata)
        """
        metadata = {
            "engine": "OpenCV" if self.use_opencv else "PIL/NumPy",
            "original_size": image.size,
            "rotation_angle": 0.0,
            "applied_steps": []
        }

        # Step 1: Ensure RGB / Grayscale
        gray_img = image.convert("L")
        metadata["applied_steps"].append("Grayscale Conversion")

        # Step 2: Deskew if requested
        if deskew:
            angle = self._calculate_skew_angle(gray_img)
            if abs(angle) > 0.5:
                gray_img = gray_img.rotate(-angle, expand=True, fillcolor=255)
                metadata["rotation_angle"] = round(angle, 2)
                metadata["applied_steps"].append(f"Deskewing ({angle:.1f}°)")

        # Step 3: Denoise
        if denoise:
            if self.use_opencv:
                np_arr = np.array(gray_img)
                denoised = cv2.medianBlur(np_arr, 3)
                gray_img = Image.fromarray(denoised)
            else:
                gray_img = gray_img.filter(ImageFilter.MedianFilter(size=3))
            metadata["applied_steps"].append("Noise Filtering")

        # Step 4: Contrast Enhancement
        if enhance_contrast:
            if self.use_opencv:
                np_arr = np.array(gray_img)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(np_arr)
                gray_img = Image.fromarray(enhanced)
            else:
                enhancer = ImageEnhance.Contrast(gray_img)
                gray_img = enhancer.enhance(1.8)
                # Auto-contrast stretch
                gray_img = ImageOps.autocontrast(gray_img, cutoff=2)
            metadata["applied_steps"].append("Contrast Enhancement")

        # Step 5: Binarization (Otsu's Thresholding)
        if binarize:
            np_arr = np.array(gray_img)
            threshold = self._otsu_threshold(np_arr)
            bin_arr = (np_arr > threshold).astype(np.uint8) * 255
            gray_img = Image.fromarray(bin_arr)
            metadata["applied_steps"].append(f"Binarization (Threshold: {threshold})")

        metadata["final_size"] = gray_img.size
        return gray_img, metadata

    def _otsu_threshold(self, gray_np: np.ndarray) -> int:
        """Calculates Otsu's optimal threshold via NumPy."""
        hist, _ = np.histogram(gray_np.ravel(), bins=256, range=(0, 256))
        total_pixels = gray_np.size
        
        current_max, threshold = 0, 127
        sum_total = np.dot(np.arange(256), hist)
        sum_back, weight_back = 0, 0
        
        for t in range(256):
            weight_back += hist[t]
            if weight_back == 0:
                continue
            weight_fore = total_pixels - weight_back
            if weight_fore == 0:
                break
                
            sum_back += t * hist[t]
            mean_back = sum_back / weight_back
            mean_fore = (sum_total - sum_back) / weight_fore
            
            var_between = weight_back * weight_fore * ((mean_back - mean_fore) ** 2)
            if var_between > current_max:
                current_max = var_between
                threshold = t
                
        return int(threshold)

    def _calculate_skew_angle(self, gray_img: Image.Image) -> float:
        """Estimates skew angle of text lines using projection profile analysis."""
        try:
            # Downscale for fast skew estimation
            small_img = gray_img.resize((min(600, gray_img.width), min(800, gray_img.height)))
            arr = np.array(small_img)
            # Binary inverse
            binary = arr < 128
            
            best_score = -1.0
            best_angle = 0.0
            # Test angles between -15 and 15 degrees in 1-degree steps
            angles = np.arange(-15.0, 15.5, 1.0)
            
            for angle in angles:
                rotated = Image.fromarray(binary).rotate(angle, expand=False, fillcolor=0)
                proj = np.sum(np.array(rotated), axis=1)
                # Score is variance of row projections (higher when text lines align horizontally)
                score = np.var(proj)
                if score > best_score:
                    best_score = score
                    best_angle = angle
                    
            return float(best_angle)
        except Exception:
            return 0.0

    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Converts PIL Image back to bytes."""
        buf = io.BytesIO()
        image.save(buf, format=format)
        return buf.getvalue()
