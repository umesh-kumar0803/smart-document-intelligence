"""
End-to-End Document Intelligence Pipeline Orchestrator.
Executes: Input Validation -> PDF/Image Processing -> OCR -> Text Preprocessing ->
Transformer Classification -> Information Extraction -> Confidence & Validation -> Structured JSON.
"""
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from PIL import Image

from app.schemas import (
    DocumentProcessResponse, DocumentMetadata, ClassificationResult,
    ValidationResult, FieldExtractionResult, ProcessingStep
)
from app.core.input_validator import InputValidator, ValidationError
from app.core.image_processor import ImageProcessor
from app.core.ocr_engine import OCREngine
from app.core.text_preprocessor import TextPreprocessor
from app.core.transformer_model import DocumentClassifier
from app.core.information_extractor import InformationExtractor
from app.core.validator import DocumentValidator


class DocumentIntelligencePipeline:
    """Orchestrates the entire smart document analysis workflow."""

    def __init__(self):
        self.validator = InputValidator()
        self.image_processor = ImageProcessor()
        self.ocr_engine = OCREngine()
        self.text_preprocessor = TextPreprocessor()
        self.classifier = DocumentClassifier(use_neural_net=True)
        self.extractor = InformationExtractor()
        self.document_validator = DocumentValidator()

    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        forced_category: Optional[str] = None
    ) -> DocumentProcessResponse:
        """
        Executes end-to-end processing of an uploaded document.
        """
        start_time = time.perf_counter()
        steps: List[ProcessingStep] = []

        # -------------------------------------------------------------
        # Stage 1: Input Validation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        is_valid, file_type, error_msg = self.validator.validate_file(file_bytes, filename)
        if not is_valid:
            raise ValidationError(error_msg)
        steps.append(ProcessingStep(
            step_name="Input Validation",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            details=f"Validated file format: {file_type}, size: {len(file_bytes):,} bytes"
        ))

        # -------------------------------------------------------------
        # Stage 2: PDF / Image Processing & OCR Engine
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if file_type == "pdf":
            ocr_result = self.ocr_engine.extract_from_pdf(file_bytes)
        else:
            ocr_result = self.ocr_engine.extract_from_image(file_bytes)

        raw_text = ocr_result.get("text", "")
        page_count = ocr_result.get("page_count", 1)
        ocr_engine_used = ocr_result.get("engine_used", "OCR Engine")
        ocr_conf = ocr_result.get("confidence", 0.90)

        steps.append(ProcessingStep(
            step_name="PDF/Image Processing & OCR Engine",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            details=f"Extracted {len(raw_text):,} characters across {page_count} page(s) using {ocr_engine_used}"
        ))

        # -------------------------------------------------------------
        # Stage 3: Text Preprocessing & Normalization
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        cleaned_text = self.text_preprocessor.clean_text(raw_text)
        stats = self.text_preprocessor.get_text_statistics(cleaned_text)

        steps.append(ProcessingStep(
            step_name="Text Preprocessing",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            details=f"Normalized text: {stats['word_count']} words, {stats['sentence_count']} sentences, vocab: {stats['vocabulary_size']}"
        ))

        # -------------------------------------------------------------
        # Stage 4: Transformer & Deep Learning Classification
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if forced_category:
            classification_data = {
                "predicted_category": forced_category,
                "category_display_name": forced_category,
                "confidence": 1.0,
                "category_scores": {forced_category: 1.0},
                "model_architecture": "User Specified Category"
            }
        else:
            classification_data = self.classifier.classify(cleaned_text)

        classification_res = ClassificationResult(**classification_data)
        category = classification_res.predicted_category

        steps.append(ProcessingStep(
            step_name="Transformer Document Classification",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            details=f"Predicted: '{classification_res.category_display_name}' (Confidence: {classification_res.confidence:.1%})"
        ))

        # -------------------------------------------------------------
        # Stage 5: Category-Specific Information Extraction
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        extracted_fields, field_details = self.extractor.extract(cleaned_text, category)

        steps.append(ProcessingStep(
            step_name="Information Extraction",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            details=f"Extracted {len(extracted_fields)} structured field(s) for category '{category}'"
        ))

        # -------------------------------------------------------------
        # Stage 6: Confidence & Validation Assurance
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        val_result = self.document_validator.validate(
            category=category,
            fields=extracted_fields,
            classification_confidence=classification_res.confidence
        )

        steps.append(ProcessingStep(
            step_name="Confidence & Validation Assurance",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            details=f"Status: {val_result.status}, Quality Score: {val_result.overall_score:.1%}, Passed: {len(val_result.passed_checks)}"
        ))

        # Total timing
        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        metadata = DocumentMetadata(
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(file_bytes),
            page_count=page_count,
            num_words=stats["word_count"],
            num_characters=stats["character_count"],
            ocr_engine_used=ocr_engine_used,
            processing_time_ms=total_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        preview = cleaned_text[:500] + ("..." if len(cleaned_text) > 500 else "")

        return DocumentProcessResponse(
            success=True,
            metadata=metadata,
            classification=classification_res,
            extracted_fields=extracted_fields,
            field_details=field_details,
            cleaned_text_preview=preview,
            raw_text=raw_text,
            validation=val_result,
            processing_steps=steps
        )
