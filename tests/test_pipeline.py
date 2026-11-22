"""
Unit and Integration Tests for Document Intelligence Pipeline.
"""
from pathlib import Path
import pytest

from app.core.input_validator import InputValidator, ValidationError
from app.core.image_processor import ImageProcessor
from app.core.ocr_engine import OCREngine
from app.core.text_preprocessor import TextPreprocessor
from app.core.transformer_model import DocumentClassifier
from app.core.information_extractor import InformationExtractor
from app.core.validator import DocumentValidator
from app.core.pipeline import DocumentIntelligencePipeline
from app.utils.sample_generator import SampleGenerator


class TestDocumentIntelligencePipeline:

    @classmethod
    def setup_class(cls):
        """Prepare sample files for testing."""
        cls.sample_gen = SampleGenerator()
        cls.samples = cls.sample_gen.generate_all_samples()
        cls.pipeline = DocumentIntelligencePipeline()

    def test_input_validator_valid_pdf(self):
        inv_pdf = Path(self.samples["Invoice"]["pdf"]).read_bytes()
        is_valid, ftype, err = InputValidator.validate_file(inv_pdf, "invoice.pdf")
        assert is_valid is True
        assert ftype == "pdf"
        assert err == ""

    def test_input_validator_empty_file(self):
        is_valid, ftype, err = InputValidator.validate_file(b"", "empty.pdf")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_input_validator_invalid_extension(self):
        is_valid, ftype, err = InputValidator.validate_file(b"test data", "test.exe")
        assert is_valid is False
        assert "unsupported" in err.lower()

    def test_text_preprocessor_cleaning(self):
        tp = TextPreprocessor()
        raw = "INVOICE   #  12345 \r\n\r\n\r\n  Total:   $500.00 \n\n"
        cleaned = tp.clean_text(raw)
        assert "INVOICE # 12345" in cleaned
        assert "$500.00" in cleaned
        stats = tp.get_text_statistics(cleaned)
        assert stats["word_count"] > 0
        assert stats["character_count"] > 0

    def test_document_classifier_invoice(self):
        clf = DocumentClassifier()
        text = "COMMERCIAL INVOICE # INV-9901 Date: 2024-05-12 Subtotal: $4,000 Tax: $400 Total Due: $4,400 Bill to Acme"
        res = clf.classify(text)
        assert res["predicted_category"] == "Invoice"
        assert res["confidence"] >= 0.50

    def test_document_classifier_resume(self):
        clf = DocumentClassifier()
        text = "Alex Mercer Senior Software Engineer Email: alex@test.com Phone: 555-0192 Skills: Python, PyTorch, Docker Education: Stanford University"
        res = clf.classify(text)
        assert res["predicted_category"] == "Resume"
        assert res["confidence"] >= 0.50

    def test_document_classifier_contract(self):
        clf = DocumentClassifier()
        text = "NON-DISCLOSURE AGREEMENT This Agreement is made between Alpha Corp and Beta LLC. Governing Law: State of Delaware. In witness whereof."
        res = clf.classify(text)
        assert res["predicted_category"] == "Contract"
        assert res["confidence"] >= 0.50

    def test_document_classifier_identity_document(self):
        clf = DocumentClassifier()
        text = "PASSPORT United States of America. Passport No: USA-994820194 Name: TAYLOR, JORDAN Date of Birth: 1994-08-22 Nationality: USA"
        res = clf.classify(text)
        assert res["predicted_category"] == "Identity_Document"
        assert res["confidence"] >= 0.50

    def test_information_extractor_invoice(self):
        extractor = InformationExtractor()
        text = "INVOICE # INV-2024-8849 Date: 2024-03-15 Due Date: 2024-04-15 Subtotal: $10,000.00 Tax: $1,000.00 Total: $11,000.00"
        fields, details = extractor.extract(text, "Invoice")
        assert fields["invoice_number"] == "INV-2024-8849"
        assert fields["subtotal"] == 10000.00
        assert fields["tax_amount"] == 1000.00
        assert fields["total_amount"] == 11000.00

    def test_validator_invoice_math_check(self):
        validator = DocumentValidator()
        fields = {
            "invoice_number": "INV-100",
            "invoice_date": "2024-01-01",
            "subtotal": 100.0,
            "tax_amount": 10.0,
            "total_amount": 110.0,
            "vendor_name": "Test Vendor",
            "customer_name": "Test Client"
        }
        res = validator.validate("Invoice", fields, 0.95)
        assert res.is_valid is True
        assert res.status == "Valid"
        assert any("reconciliation verified" in chk for chk in res.passed_checks)

    def test_end_to_end_pipeline_invoice_pdf(self):
        inv_pdf = Path(self.samples["Invoice"]["pdf"]).read_bytes()
        res = self.pipeline.process_document(inv_pdf, "sample_invoice.pdf")
        assert res.success is True
        assert res.classification.predicted_category == "Invoice"
        assert res.metadata.page_count == 1
        assert res.metadata.num_words > 20
        assert len(res.processing_steps) >= 5
        assert res.validation.status in ["Valid", "Requires Review"]

    def test_end_to_end_pipeline_resume_pdf(self):
        res_pdf = Path(self.samples["Resume"]["pdf"]).read_bytes()
        res = self.pipeline.process_document(res_pdf, "sample_resume.pdf")
        assert res.success is True
        assert res.classification.predicted_category == "Resume"
        assert "alex.mercer@gmail.com" in res.extracted_fields.get("email", "")
        assert len(res.extracted_fields.get("skills", [])) >= 3
