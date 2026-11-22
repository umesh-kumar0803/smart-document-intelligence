"""
Tests for FastAPI endpoints.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main_api import app
from app.utils.sample_generator import SampleGenerator

client = TestClient(app)


class TestFastAPIEndpoints:

    @classmethod
    def setup_class(cls):
        cls.sample_gen = SampleGenerator()
        cls.samples = cls.sample_gen.generate_all_samples()

    def test_root_endpoint(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "Invoice" in data["supported_categories"]

    def test_health_endpoint(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert len(data["loaded_models"]) > 0

    def test_categories_endpoint(self):
        resp = client.get("/api/v1/categories")
        assert resp.status_code == 200
        cats = resp.json()["categories"]
        assert len(cats) >= 6

    def test_classify_text_endpoint(self):
        payload = {"text": "COMMERCIAL INVOICE # 4920 Date: 2024-03-01 Total Due: $1,200.00 Subtotal: $1,100 Tax: $100"}
        resp = client.post("/api/v1/documents/classify", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_category"] == "Invoice"
        assert data["confidence"] > 0.50

    def test_extract_text_endpoint(self):
        payload = {"text": "COMMERCIAL INVOICE # INV-990 Date: 2024-03-01 Total: $1,200.00 Subtotal: $1,100 Tax: $100", "category": "Invoice"}
        resp = client.post("/api/v1/documents/extract", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "Invoice"
        assert data["extracted_fields"]["invoice_number"] == "INV-990"

    def test_process_document_endpoint_invoice(self):
        pdf_path = Path(self.samples["Invoice"]["pdf"])
        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/api/v1/documents/process",
                files={"file": ("sample_invoice.pdf", f, "application/pdf")}
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["classification"]["predicted_category"] == "Invoice"
        assert data["extracted_fields"]["invoice_number"] == "INV-2024-8849"
        assert data["extracted_fields"]["total_amount"] == 11000.00

    def test_batch_process_endpoint(self):
        pdf1 = Path(self.samples["Invoice"]["pdf"])
        pdf2 = Path(self.samples["Resume"]["pdf"])
        with open(pdf1, "rb") as f1, open(pdf2, "rb") as f2:
            resp = client.post(
                "/api/v1/documents/batch",
                files=[
                    ("files", ("invoice.pdf", f1, "application/pdf")),
                    ("files", ("resume.pdf", f2, "application/pdf"))
                ]
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 2
        assert data["successful_count"] == 2

    def test_metrics_endpoint(self):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_processed_documents"] > 0

    def test_evaluation_endpoint(self):
        resp = client.get("/api/v1/evaluation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["accuracy"] > 0.80
        assert "confusion_matrix" in data
