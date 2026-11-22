"""
Input validation module for uploaded documents (PDFs and Images).
"""
from pathlib import Path
from typing import Tuple
from app.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES


class ValidationError(Exception):
    """Custom exception raised when document validation fails."""
    pass


class InputValidator:
    """Validates document format, size, extensions, and structural integrity."""

    @staticmethod
    def validate_file(file_bytes: bytes, filename: str) -> Tuple[bool, str, str]:
        """
        Validates the incoming file bytes and filename.
        
        Returns:
            Tuple[bool, str, str]: (is_valid, detected_file_type, error_message)
        """
        if not file_bytes or len(file_bytes) == 0:
            return False, "", "The uploaded file is empty (0 bytes)."

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            actual_mb = len(file_bytes) / (1024 * 1024)
            return False, "", f"File size ({actual_mb:.2f} MB) exceeds maximum allowed size ({max_mb:.0f} MB)."

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(SUPPORTED_EXTENSIONS)
            return False, "", f"Unsupported file extension '{ext}'. Supported formats: {allowed}."

        # Magic number / byte signature validation
        file_type = "unknown"
        if file_bytes.startswith(b"%PDF-"):
            file_type = "pdf"
        elif file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            file_type = "image/png"
        elif file_bytes.startswith(b"\xff\xd8\xff"):
            file_type = "image/jpeg"
        elif file_bytes.startswith(b"II*\x00") or file_bytes.startswith(b"MM\x00*"):
            file_type = "image/tiff"
        elif file_bytes.startswith(b"BM"):
            file_type = "image/bmp"
        elif file_bytes.startswith(b"RIFF") and len(file_bytes) > 12 and file_bytes[8:12] == b"WEBP":
            file_type = "image/webp"
        else:
            # Check by extension if magic check is relaxed
            if ext == ".pdf":
                file_type = "pdf"
            elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
                file_type = f"image/{ext.replace('.', '')}"
            else:
                return False, "", "File signature does not match any supported PDF or Image format."

        return True, file_type, ""
