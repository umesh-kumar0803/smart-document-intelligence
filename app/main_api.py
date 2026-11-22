"""
FastAPI REST API Application for Smart Document Intelligence.
Exposes endpoints for document upload, OCR extraction, transformer classification,
structured entity extraction, batch processing, and live evaluation.
"""
import time
import os
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    CATEGORIES, CATEGORY_DISPLAY_NAMES, CATEGORY_KEYWORDS, SAMPLE_DIR
)
from app.schemas import (
    DocumentProcessResponse, BatchProcessResponse, ClassifyTextRequest,
    ExtractTextRequest, HealthResponse, MetricsResponse, ClassificationResult,
    DocumentMetadata, ValidationResult
)
from app.core.pipeline import DocumentIntelligencePipeline
from app.core.input_validator import ValidationError
from evaluation.evaluator import SystemEvaluator

# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend applications (Streamlit, React, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance & metrics tracking
pipeline = DocumentIntelligencePipeline()
evaluator = SystemEvaluator(pipeline.classifier)

system_metrics = {
    "total_processed_documents": 0,
    "total_processing_time_ms": 0.0,
    "documents_by_category": {cat: 0 for cat in CATEGORIES},
    "documents_by_category_unknown": 0,
    "ocr_mode_breakdown": {}
}


@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service overview and documentation links."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "operational",
        "interactive_docs": "/docs",
        "redoc_docs": "/redoc",
        "supported_categories": CATEGORIES
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System Health"])
async def health_check():
    """Returns application health, hardware device status, and loaded models."""
    import sys
    import platform
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tess_path = pipeline.ocr_engine.tesseract_cmd

    return HealthResponse(
        status="healthy",
        system=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        device=f"PyTorch Device: {device}",
        tesseract_available=pipeline.ocr_engine.tesseract_available,
        tesseract_path=tess_path,
        loaded_models=[
            pipeline.classifier.model_name,
            "PyMuPDF 1.28.2 (fitz) Vector Engine",
            "Scikit-learn 1.9.0 N-Gram Vectorizer",
            "Rule-Based Regex & Entity Taxonomy Engine"
        ],
        supported_categories=CATEGORIES
    )


@app.get("/api/v1/categories", tags=["Document Schema"])
async def get_categories():
    """Lists supported document categories, human-readable display names, and key trigger words."""
    return {
        "categories": [
            {
                "category_id": cat,
                "display_name": CATEGORY_DISPLAY_NAMES.get(cat, cat),
                "key_indicators": CATEGORY_KEYWORDS.get(cat, [])
            }
            for cat in CATEGORIES
        ]
    }


@app.post(
    "/api/v1/documents/process",
    response_model=DocumentProcessResponse,
    tags=["Document Processing"]
)
async def process_document(
    file: UploadFile = File(..., description="Document file to process (PDF, PNG, JPG, TIFF)"),
    forced_category: Optional[str] = Form(None, description="Optional override for document category")
):
    """
    Primary processing endpoint:
    Accepts PDF or Image -> OCR -> Preprocessing -> Transformer Classification -> Information Extraction -> Validation.
    """
    try:
        content = await file.read()
        response = pipeline.process_document(
            file_bytes=content,
            filename=file.filename or "uploaded_document",
            forced_category=forced_category
        )

        # Update metrics
        system_metrics["total_processed_documents"] += 1
        system_metrics["total_processing_time_ms"] += response.metadata.processing_time_ms
        cat = response.classification.predicted_category
        if cat in system_metrics["documents_by_category"]:
            system_metrics["documents_by_category"][cat] += 1
        else:
            system_metrics["documents_by_category_unknown"] += 1

        ocr_engine = response.metadata.ocr_engine_used
        system_metrics["ocr_mode_breakdown"][ocr_engine] = (
            system_metrics["ocr_mode_breakdown"].get(ocr_engine, 0) + 1
        )

        return response

    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")


@app.post(
    "/api/v1/documents/batch",
    response_model=BatchProcessResponse,
    tags=["Document Processing"]
)
async def batch_process_documents(
    files: List[UploadFile] = File(..., description="Multiple document files to process simultaneously")
):
    """
    Batch processing endpoint for uploading and analyzing multiple documents in a single transaction.
    """
    start_time = time.perf_counter()
    results: List[DocumentProcessResponse] = []
    success_count = 0
    fail_count = 0

    for file in files:
        try:
            content = await file.read()
            resp = pipeline.process_document(
                file_bytes=content,
                filename=file.filename or "batch_file"
            )
            results.append(resp)
            success_count += 1
            system_metrics["total_processed_documents"] += 1
            system_metrics["total_processing_time_ms"] += resp.metadata.processing_time_ms
            cat = resp.classification.predicted_category
            if cat in system_metrics["documents_by_category"]:
                system_metrics["documents_by_category"][cat] += 1
        except Exception:
            fail_count += 1

    total_batch_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return BatchProcessResponse(
        total_documents=len(files),
        successful_count=success_count,
        failed_count=fail_count,
        total_processing_time_ms=total_batch_time_ms,
        results=results
    )


@app.post(
    "/api/v1/documents/classify",
    response_model=ClassificationResult,
    tags=["NLP & Classification"]
)
async def classify_raw_text(request: ClassifyTextRequest):
    """Classifies raw text directly using the Transformer deep learning representation."""
    cleaned = pipeline.text_preprocessor.clean_text(request.text)
    result = pipeline.classifier.classify(cleaned)
    return ClassificationResult(**result)


@app.post(
    "/api/v1/documents/extract",
    tags=["NLP & Classification"]
)
async def extract_information_from_text(request: ExtractTextRequest):
    """Extracts structured entities and fields from raw text given an optional or predicted category."""
    cleaned = pipeline.text_preprocessor.clean_text(request.text)
    category = request.category
    if not category:
        clf = pipeline.classifier.classify(cleaned)
        category = clf["predicted_category"]

    fields, details = pipeline.extractor.extract(cleaned, category)
    val = pipeline.document_validator.validate(category, fields, 1.0)

    return {
        "category": category,
        "extracted_fields": fields,
        "field_details": [d.model_dump() for d in details],
        "validation": val.model_dump()
    }


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Analytics"])
async def get_metrics():
    """Returns runtime telemetry, processing counts, and average latencies."""
    total_docs = system_metrics["total_processed_documents"]
    avg_latency = (
        round(system_metrics["total_processing_time_ms"] / total_docs, 2)
        if total_docs > 0 else 0.0
    )
    return MetricsResponse(
        total_processed_documents=total_docs,
        average_processing_time_ms=avg_latency,
        documents_by_category=system_metrics["documents_by_category"],
        ocr_mode_breakdown=system_metrics["ocr_mode_breakdown"]
    )


@app.get("/api/v1/evaluation", tags=["Analytics"])
async def run_system_evaluation():
    """Runs on-demand scikit-learn evaluation metrics against the benchmark test set."""
    return evaluator.evaluate_test_set()


@app.get("/api/v1/samples/{category}", tags=["Sample Generator"])
async def get_sample_document(
    category: str,
    format: str = Query("pdf", enum=["pdf", "png"])
):
    """Fetches a pre-generated realistic sample PDF or PNG document for testing."""
    ext = ".pdf" if format.lower() == "pdf" else ".png"
    cat_lower = category.lower().replace(" ", "_")
    target_filename = f"sample_{cat_lower}{ext}"
    target_path = SAMPLE_DIR / target_filename

    if not target_path.exists():
        # Regenerate if missing
        from app.utils.sample_generator import SampleGenerator
        gen = SampleGenerator()
        gen.generate_all_samples()

    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample document for category '{category}' not found.")

    media_type = "application/pdf" if ext == ".pdf" else "image/png"
    return FileResponse(path=str(target_path), filename=target_filename, media_type=media_type)
