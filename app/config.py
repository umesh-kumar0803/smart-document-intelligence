"""
Configuration settings for the Smart Document Intelligence System.
"""
import os
from pathlib import Path
from typing import Dict, List, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_documents"
UPLOAD_DIR = BASE_DIR / "uploads"

# Ensure directories exist
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Document Categories
CATEGORIES: List[str] = [
    "Invoice",
    "Resume",
    "Contract",
    "Identity_Document",
    "Financial_Report",
    "Research_Report",
    "Receipt",
]

CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    "Invoice": "Commercial Invoice / Bill",
    "Resume": "Candidate Resume / CV",
    "Contract": "Legal Contract / Agreement",
    "Identity_Document": "Identity Document / Passport / ID",
    "Financial_Report": "Financial & Earnings Report",
    "Research_Report": "Technical & Research Report",
    "Receipt": "Sales / Expense Receipt",
}

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Invoice": [
        "invoice", "bill to", "invoice number", "inv-", "subtotal", "tax", "due date", 
        "amount due", "payment terms", "balance due", "unit price", "purchase order", "po number"
    ],
    "Resume": [
        "resume", "curriculum vitae", "experience", "education", "skills", "objective", 
        "summary", "work history", "bachelor", "master", "phd", "university", "technologies", "projects"
    ],
    "Contract": [
        "agreement", "contract", "parties", "hereby", "confidentiality", "terms and conditions",
        "governing law", "jurisdiction", "indemnification", "termination", "effective date", "witnesseth"
    ],
    "Identity_Document": [
        "passport", "national identity", "driver license", "driving licence", "identity card", 
        "date of birth", "dob", "nationality", "place of birth", "sex", "id no", "expiry date", "issuing authority"
    ],
    "Financial_Report": [
        "financial statement", "balance sheet", "income statement", "cash flow", "fiscal year", 
        "quarterly report", "revenue", "net income", "ebitda", "operating margin", "shareholders", "q1", "q2", "q3", "q4"
    ],
    "Research_Report": [
        "abstract", "methodology", "experimental results", "conclusion", "references", 
        "introduction", "literature review", "findings", "hypothesis", "dataset", "evaluation metrics"
    ],
    "Receipt": [
        "receipt", "cash receipt", "sales receipt", "terminal", "visa", "mastercard", "change due",
        "cashier", "items sold", "order #", "merchant", "store #", "vat reg"
    ]
}

# Image Processing Configuration
IMAGE_DPI = 300
CONTRAST_ENHANCE_FACTOR = 1.6
SHARPNESS_FACTOR = 1.3
SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

# OCR Paths to probe on Windows
TESSERACT_CANDIDATE_PATHS = [
    os.environ.get("TESSERACT_CMD", ""),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\HP\AppData\Local\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
]

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.50

# API Settings
API_TITLE = "Smart Document Intelligence System API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "An AI-powered application for extracting, understanding and classifying "
    "information from documents (PDFs and Images) using OCR, NLP, and Deep Learning."
)
HOST = "0.0.0.0"
PORT = 8000
