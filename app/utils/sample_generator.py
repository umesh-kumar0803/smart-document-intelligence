"""
Synthetic Sample Document Generator.
Generates realistic PDFs and Images using ReportLab and PIL for immediate testing.
"""
from pathlib import Path
from typing import Dict
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pymupdf

from app.config import SAMPLE_DIR


class SampleGenerator:
    """Generates realistic sample documents (PDF and Image) across all supported categories."""

    def __init__(self, output_dir: Path = SAMPLE_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_samples(self) -> Dict[str, Dict[str, str]]:
        """Generates sample PDFs and renders PNG equivalents for all core categories."""
        generated = {}
        generators = [
            ("Invoice", self.generate_invoice_pdf),
            ("Resume", self.generate_resume_pdf),
            ("Contract", self.generate_contract_pdf),
            ("Identity_Document", self.generate_id_pdf),
            ("Financial_Report", self.generate_financial_pdf),
            ("Research_Report", self.generate_research_pdf),
            ("Receipt", self.generate_receipt_pdf),
        ]

        for category, gen_func in generators:
            pdf_path = gen_func()
            png_path = self._convert_pdf_to_png(pdf_path)
            generated[category] = {
                "pdf": str(pdf_path),
                "png": str(png_path)
            }
        return generated

    def generate_invoice_pdf(self) -> Path:
        """Generates a professional corporate invoice PDF."""
        file_path = self.output_dir / "sample_invoice.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        width, height = letter

        # Header Banner
        c.setFillColorRGB(0.12, 0.23, 0.38)  # Deep Navy
        c.rect(0, height - 90, width, 90, fill=True, stroke=False)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(40, height - 55, "COMMERCIAL TAX INVOICE")

        # Vendor Details
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, height - 120, "Apex Cloud Technologies Ltd.")
        c.setFont("Helvetica", 10)
        c.drawString(40, height - 135, "100 Silicon Boulevard, Suite 500")
        c.drawString(40, height - 150, "San Francisco, CA 94107, USA")
        c.drawString(40, height - 165, "Email: billing@apexcloud.io | Tax ID: US-88291039")

        # Invoice Metadata Box
        c.setFont("Helvetica-Bold", 10)
        c.drawString(380, height - 120, "Invoice Number: INV-2024-8849")
        c.drawString(380, height - 135, "Invoice Date: 2024-03-15")
        c.drawString(380, height - 150, "Due Date: 2024-04-15")
        c.drawString(380, height - 165, "Payment Terms: Net 30 days")

        # Billed To
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, height - 200, "BILLED TO:")
        c.setFont("Helvetica", 10)
        c.drawString(40, height - 215, "Acme Global Innovations Inc.")
        c.drawString(40, height - 230, "Attn: Accounts Payable")
        c.drawString(40, height - 245, "550 Enterprise Way, Boston, MA 02110")

        # Table Header
        c.setFillColorRGB(0.9, 0.92, 0.95)
        c.rect(40, height - 290, width - 80, 25, fill=True, stroke=False)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, height - 275, "Description")
        c.drawString(320, height - 275, "Qty")
        c.drawString(380, height - 275, "Unit Price ($)")
        c.drawString(480, height - 275, "Total ($)")

        # Table Items
        items = [
            ("Cloud Infrastructure Architecture & Migration", "1", "4,500.00", "4,500.00"),
            ("Kubernetes Cluster Deployment & CI/CD", "2", "1,200.00", "2,400.00"),
            ("AI Pipeline & Transformer Model Serving", "1", "3,100.00", "3,100.00"),
        ]

        y = height - 315
        c.setFont("Helvetica", 10)
        for desc, qty, rate, total in items:
            c.drawString(50, y, desc)
            c.drawString(325, y, qty)
            c.drawString(390, y, rate)
            c.drawString(480, y, total)
            c.setStrokeColorRGB(0.85, 0.85, 0.85)
            c.line(40, y - 8, width - 40, y - 8)
            y -= 25

        # Totals Section
        c.setFont("Helvetica", 10)
        c.drawString(380, y - 20, "Subtotal:")
        c.drawString(480, y - 20, "$10,000.00")

        c.drawString(380, y - 40, "Sales Tax (10%):")
        c.drawString(480, y - 40, "$1,000.00")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(380, y - 65, "Total Amount Due:")
        c.drawString(480, y - 65, "$11,000.00")

        # Footer notes
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(40, 60, "Thank you for your business. Remit payment via wire transfer to Apex Cloud Technologies.")
        c.drawString(40, 45, "Bank: Chase Manhattan Bank | Routing: 021000021 | Account: 9948201948")

        c.save()
        return file_path

    def generate_resume_pdf(self) -> Path:
        """Generates a modern tech software engineer resume."""
        file_path = self.output_dir / "sample_resume.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        width, height = letter

        # Name and Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, height - 50, "Alex Mercer")
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0.2, 0.4, 0.7)
        c.drawString(40, height - 68, "Lead AI & Deep Learning Systems Engineer")

        # Contact Info Bar
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica", 9)
        c.drawString(40, height - 88, "Email: alex.mercer@gmail.com | Phone: +1 (415) 555-0192 | San Francisco, CA")
        c.drawString(40, height - 100, "LinkedIn: linkedin.com/in/alexmercer-ai | GitHub: github.com/alexmercer")

        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(40, height - 110, width - 40, height - 110)

        # Section: Summary
        y = height - 130
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.1, 0.2, 0.4)
        c.drawString(40, y, "PROFESSIONAL SUMMARY")
        c.setFont("Helvetica", 9.5)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        y -= 16
        c.drawString(40, y, "Accomplished Machine Learning Engineer with 7+ years of experience architecting large-scale deep learning")
        y -= 14
        c.drawString(40, y, "systems, computer vision pipelines, and transformer-based document intelligence platforms.")

        # Section: Skills
        y -= 25
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.1, 0.2, 0.4)
        c.drawString(40, y, "CORE COMPETENCIES & TECHNICAL SKILLS")
        y -= 16
        c.setFont("Helvetica", 9.5)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(40, y, "Languages & Frameworks: Python, PyTorch, TensorFlow, Keras, Hugging Face Transformers, FastAPI")
        y -= 14
        c.drawString(40, y, "Data & NLP: Scikit-learn, Pandas, NumPy, OpenCV, spaCy, NLTK, PyMuPDF, Tesseract OCR")
        y -= 14
        c.drawString(40, y, "Infrastructure & Cloud: Docker, Kubernetes, AWS, GCP, CI/CD, Git, Linux, PostgreSQL")

        # Section: Experience
        y -= 25
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.1, 0.2, 0.4)
        c.drawString(40, y, "PROFESSIONAL EXPERIENCE")
        
        y -= 18
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(40, y, "Senior Machine Learning Engineer | NeuralScale Inc.")
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(420, y, "2021 - Present")
        y -= 14
        c.setFont("Helvetica", 9.5)
        c.drawString(50, y, "• Deployed multi-modal document intelligence pipeline processing 500k+ documents monthly.")
        y -= 14
        c.drawString(50, y, "• Improved classification accuracy by 18% using fine-tuned Transformer encoder representations.")

        y -= 20
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(40, y, "Software Engineer, Computer Vision | Apex Systems")
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(420, y, "2018 - 2021")
        y -= 14
        c.setFont("Helvetica", 9.5)
        c.drawString(50, y, "• Developed automated OCR deskewing and adaptive thresholding algorithms with OpenCV.")
        y -= 14
        c.drawString(50, y, "• Reduced image extraction latency by 40% via vectorized NumPy routines.")

        # Section: Education
        y -= 25
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.1, 0.2, 0.4)
        c.drawString(40, y, "EDUCATION & CREDENTIALS")
        y -= 18
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(40, y, "Master of Science in Computer Science (Artificial Intelligence)")
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(420, y, "2016 - 2018")
        y -= 14
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "Stanford University, Stanford, CA")

        y -= 18
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Bachelor of Science in Software Engineering")
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(420, y, "2012 - 2016")
        y -= 14
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "University of California, Berkeley")

        c.save()
        return file_path

    def generate_contract_pdf(self) -> Path:
        """Generates a mutual Non-Disclosure Agreement (NDA)."""
        file_path = self.output_dir / "sample_contract.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2.0, height - 50, "MUTUAL NON-DISCLOSURE AGREEMENT")
        
        c.setFont("Helvetica", 10)
        y = height - 90
        c.drawString(40, y, "This Mutual Non-Disclosure Agreement ('Agreement') is made and entered into as of March 1, 2024")
        y -= 16
        c.drawString(40, y, "('Effective Date'), by and between:")
        
        y -= 20
        c.setFont("Helvetica-Bold", 10)
        c.drawString(60, y, "1. Innovatech Global Corp, a Delaware corporation ('Disclosing Party'), and")
        y -= 16
        c.drawString(60, y, "2. Quantum Logic Solutions LLC, a California limited liability company ('Receiving Party').")

        y -= 25
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "1. Purpose and Confidential Information")
        y -= 16
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "The parties wish to explore a business opportunity concerning AI model deployments. 'Confidential Information'")
        y -= 14
        c.drawString(40, y, "means any proprietary technical, financial, or operational data disclosed by either party.")

        y -= 20
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "2. Term and Termination")
        y -= 16
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "This Agreement shall remain in full force for a period of three (3) years from the Effective Date.")
        y -= 14
        c.drawString(40, y, "Either party may terminate discussions at any time upon thirty (30) days written notice.")

        y -= 20
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "3. Governing Law and Dispute Resolution")
        y -= 16
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware,")
        y -= 14
        c.drawString(40, y, "without regard to its conflict of law principles. Any arbitration shall occur in Wilmington, DE.")

        y -= 30
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "IN WITNESS WHEREOF, the authorized representatives have executed this Agreement.")
        
        y -= 40
        c.drawString(40, y, "For Innovatech Global Corp:")
        c.drawString(320, y, "For Quantum Logic Solutions LLC:")
        y -= 30
        c.line(40, y, 220, y)
        c.line(320, y, 500, y)
        y -= 15
        c.setFont("Helvetica", 9)
        c.drawString(40, y, "Signature: Sarah Jenkins, Chief Executive Officer")
        c.drawString(320, y, "Signature: David Miller, Managing Partner")

        c.save()
        return file_path

    def generate_id_pdf(self) -> Path:
        """Generates an official-looking Identity Card / Passport document."""
        file_path = self.output_dir / "sample_id_card.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        width, height = letter

        # Card Container
        c.setStrokeColorRGB(0.2, 0.4, 0.7)
        c.setLineWidth(2)
        c.roundRect(60, height - 320, 480, 260, 10, stroke=True, fill=False)

        # Header Ribbon
        c.setFillColorRGB(0.12, 0.24, 0.45)
        c.rect(61, height - 120, 478, 59, fill=True, stroke=False)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(80, height - 90, "UNITED STATES OF AMERICA")
        c.setFont("Helvetica", 11)
        c.drawString(80, height - 108, "OFFICIAL IDENTITY DOCUMENT / PASSPORT CARD")

        # Photo placeholder
        c.setStrokeColorRGB(0.5, 0.5, 0.5)
        c.setFillColorRGB(0.92, 0.92, 0.92)
        c.rect(80, height - 260, 110, 130, fill=True, stroke=True)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.setFont("Helvetica", 10)
        c.drawCentredString(135, height - 200, "[ PHOTO ]")

        # Details
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(210, height - 145, "Document ID / Passport No:")
        c.setFont("Helvetica", 10)
        c.drawString(360, height - 145, "USA-994820194")

        c.setFont("Helvetica-Bold", 9)
        c.drawString(210, height - 165, "Full Name:")
        c.setFont("Helvetica", 10)
        c.drawString(360, height - 165, "JORDAN ELIZABETH TAYLOR")

        c.setFont("Helvetica-Bold", 9)
        c.drawString(210, height - 185, "Date of Birth (DOB):")
        c.setFont("Helvetica", 10)
        c.drawString(360, height - 185, "1994-08-22")

        c.setFont("Helvetica-Bold", 9)
        c.drawString(210, height - 205, "Nationality:")
        c.setFont("Helvetica", 10)
        c.drawString(360, height - 205, "UNITED STATES")

        c.setFont("Helvetica-Bold", 9)
        c.drawString(210, height - 225, "Sex / Gender:")
        c.setFont("Helvetica", 10)
        c.drawString(360, height - 225, "FEMALE (F)")

        c.setFont("Helvetica-Bold", 9)
        c.drawString(210, height - 245, "Expiry Date:")
        c.setFont("Helvetica-Bold", 10)
        c.drawString(360, height - 245, "2034-08-21")

        # MRZ Machine Readable Zone
        c.setFont("Courier-Bold", 9.5)
        c.drawString(80, height - 285, "P<USATAYLOR<<JORDAN<ELIZABETH<<<<<<<<<<<<<<<<<")
        c.drawString(80, height - 302, "9948201940USA9408224F3408218<<<<<<<<<<<<<<<04")

        c.save()
        return file_path

    def generate_financial_pdf(self) -> Path:
        """Generates an executive quarterly financial earnings report."""
        file_path = self.output_dir / "sample_financial_report.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, height - 50, "GLOBAL FINTECH CORP - Q3 2024 FINANCIAL REPORT")
        c.setFont("Helvetica", 11)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawString(40, height - 68, "Quarter Ended September 30, 2024 | Published October 18, 2024")

        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(40, height - 78, width - 40, height - 78)

        y = height - 105
        c.setFillColorRGB(0.1, 0.2, 0.4)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "1. Executive Summary & Operational Highlights")
        
        y -= 18
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "Global FinTech Corp delivered another record quarter, driven by enterprise cloud adoption and payment volume.")
        y -= 14
        c.drawString(40, y, "Total Revenue reached $5.8 Billion, an increase of 16% year-over-year. Operating Income rose to $1.6 Billion.")

        # Key Metrics Table
        y -= 30
        c.setFillColorRGB(0.92, 0.95, 0.98)
        c.rect(40, y - 90, width - 80, 90, fill=True, stroke=False)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(55, y - 20, "Key Financial Metric")
        c.drawString(280, y - 20, "Q3 2024")
        c.drawString(400, y - 20, "YoY Growth")

        metrics = [
            ("Total Gross Revenue", "$5.80 Billion", "+16.2%"),
            ("Net Income", "$1.25 Billion", "+21.4%"),
            ("Operating Margin", "27.6%", "+180 bps"),
            ("Free Cash Flow", "$1.40 Billion", "+14.8%"),
        ]

        my = y - 38
        c.setFont("Helvetica", 9.5)
        for name, val, growth in metrics:
            c.drawString(55, my, name)
            c.drawString(280, my, val)
            c.drawString(400, my, growth)
            my -= 16

        c.save()
        return file_path

    def generate_receipt_pdf(self) -> Path:
        """Generates a retail POS store receipt."""
        file_path = self.output_dir / "sample_receipt.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        width, height = letter

        # Narrow receipt simulation
        rx = width / 2.0 - 150
        rw = 300
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.rect(rx, height - 450, rw, 400, fill=False, stroke=True)

        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2.0, height - 80, "METRO URBAN MARKET")
        c.setFont("Helvetica", 9)
        c.drawCentredString(width / 2.0, height - 95, "Store # 4091 - 14th Ave, New York NY")
        c.drawCentredString(width / 2.0, height - 108, "Tel: (212) 555-8930 | Date: 2024-03-22 13:42")
        c.drawCentredString(width / 2.0, height - 120, "Cashier: Samantha | Order # 99420")

        c.line(rx + 15, height - 130, rx + rw - 15, height - 130)

        # Line items
        items = [
            ("Organic Cold Brew Coffee", "$4.95"),
            ("Artisan Sourdough Loaf", "$5.50"),
            ("Greek Yogurt 32oz", "$6.25"),
            ("Fresh Hass Avocados (x4)", "$5.00"),
        ]

        y = height - 150
        for name, price in items:
            c.drawString(rx + 20, y, name)
            c.drawRightString(rx + rw - 20, y, price)
            y -= 20

        c.line(rx + 15, y - 5, rx + rw - 15, y - 5)
        y -= 22

        c.drawString(rx + 20, y, "Subtotal:")
        c.drawRightString(rx + rw - 20, y, "$21.70")
        y -= 16

        c.drawString(rx + 20, y, "Sales Tax (8.875%):")
        c.drawRightString(rx + rw - 20, y, "$1.93")
        y -= 22

        c.setFont("Helvetica-Bold", 11)
        c.drawString(rx + 20, y, "TOTAL AMOUNT:")
        c.drawRightString(rx + rw - 20, y, "$23.63")
        y -= 25

        c.setFont("Helvetica", 9)
        c.drawString(rx + 20, y, "Payment Method: VISA **********8942")
        y -= 14
        c.drawString(rx + 20, y, "Auth Code: 83921 | Status: APPROVED")
        y -= 25
        c.setFont("Helvetica-Oblique", 8.5)
        c.drawCentredString(width / 2.0, y, "Thank you for shopping at Metro Urban Market!")

        c.save()
        return file_path

    def generate_research_pdf(self) -> Path:
        """Generates a technical scientific research report."""
        file_path = self.output_dir / "sample_research_report.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2.0, height - 50, "DEEP LEARNING ARCHITECTURES FOR DOCUMENT INTELLIGENCE")
        c.setFont("Helvetica", 10)
        c.drawCentredString(width / 2.0, height - 68, "Dr. Elena Rostova & Dr. Alex Mercer | Department of AI Systems, MIT")
        
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(40, height - 80, width - 40, height - 80)

        y = height - 105
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "ABSTRACT")
        y -= 16
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "Extracting structured semantics from unstructured visual documents remains a crucial capability in enterprise NLP.")
        y -= 14
        c.drawString(40, y, "In this paper, we explore multi-modal Transformer representations combined with localized OCR feature embeddings.")
        y -= 14
        c.drawString(40, y, "Experimental results demonstrate that our architecture attains 95.2% classification accuracy and 95.1% weighted F1-score.")

        y -= 25
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "1. INTRODUCTION & METHODOLOGY")
        y -= 16
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "Traditional OCR engines often fail when document layouts vary across enterprise forms, contracts, and invoices.")
        y -= 14
        c.drawString(40, y, "We propose an end-to-end pipeline that combines adaptive computer vision preprocessing with deep neural classifiers.")
        y -= 14
        c.drawString(40, y, "Our evaluation on the holdout benchmark dataset confirms robust generalization across heterogeneous document types.")

        y -= 25
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "2. EXPERIMENTAL RESULTS & CONCLUSION")
        y -= 16
        c.setFont("Helvetica", 9.5)
        c.drawString(40, y, "The proposed Transformer MLP model achieved superior sub-second inference latency (~250 ms) without requiring external GPU.")
        y -= 14
        c.drawString(40, y, "We conclude that hybrid semantic fusion provides high precision for automated document workflows and structured extraction.")

        c.save()
        return file_path

    def _convert_pdf_to_png(self, pdf_path: Path) -> Path:
        """Renders first page of PDF as high-resolution PNG image."""
        png_path = pdf_path.with_suffix(".png")
        doc = pymupdf.open(str(pdf_path))
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        pix.save(str(png_path))
        doc.close()
        return png_path
