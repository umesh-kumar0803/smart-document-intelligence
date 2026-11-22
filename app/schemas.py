"""
Pydantic Schemas for Request and Response validation.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DocumentCategory(str, Enum):
    INVOICE = "Invoice"
    RESUME = "Resume"
    CONTRACT = "Contract"
    IDENTITY_DOCUMENT = "Identity_Document"
    FINANCIAL_REPORT = "Financial_Report"
    RESEARCH_REPORT = "Research_Report"
    RECEIPT = "Receipt"
    UNKNOWN = "Unknown"


class FieldExtractionResult(BaseModel):
    field_name: str
    value: Any = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    data_type: str = "string"
    is_valid: bool = True
    validation_notes: Optional[str] = None


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int = 1
    num_words: int = 0
    num_characters: int = 0
    ocr_engine_used: str = "PyMuPDF / OCR"
    processing_time_ms: float = 0.0
    timestamp: str


class ClassificationResult(BaseModel):
    predicted_category: str
    category_display_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    category_scores: Dict[str, float]
    model_architecture: str = "Pretrained Transformer / Deep Learning Ensemble"


class ValidationResult(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    is_valid: bool
    status: str  # "Valid", "Requires Review", "Invalid"
    passed_checks: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []


class ProcessingStep(BaseModel):
    step_name: str
    duration_ms: float
    status: str = "completed"
    details: Optional[str] = None


class DocumentProcessResponse(BaseModel):
    success: bool
    metadata: DocumentMetadata
    classification: ClassificationResult
    extracted_fields: Dict[str, Any]
    field_details: List[FieldExtractionResult]
    cleaned_text_preview: str
    raw_text: str
    validation: ValidationResult
    processing_steps: List[ProcessingStep] = []


class BatchProcessResponse(BaseModel):
    total_documents: int
    successful_count: int
    failed_count: int
    total_processing_time_ms: float
    results: List[DocumentProcessResponse]


class ClassifyTextRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Document raw text content to classify")


class ExtractTextRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Document raw text content")
    category: Optional[str] = Field(None, description="Optional forced category for targeted extraction")


class HealthResponse(BaseModel):
    status: str
    system: str
    python_version: str
    device: str
    tesseract_available: bool
    tesseract_path: Optional[str]
    loaded_models: List[str]
    supported_categories: List[str]


class MetricsResponse(BaseModel):
    total_processed_documents: int
    average_processing_time_ms: float
    documents_by_category: Dict[str, int]
    ocr_mode_breakdown: Dict[str, int]
