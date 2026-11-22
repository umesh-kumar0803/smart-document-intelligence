"""
Evaluation module using Scikit-learn.
Computes Precision, Recall, F1-Score, Confusion Matrix, and Accuracy across document categories.
"""
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from app.config import CATEGORIES
from app.core.transformer_model import DocumentClassifier


class SystemEvaluator:
    """Evaluates Document Intelligence classification and extraction performance."""

    def __init__(self, classifier: DocumentClassifier = None):
        self.classifier = classifier or DocumentClassifier()
        self.categories = CATEGORIES

    def evaluate_test_set(self, test_texts: List[str] = None, true_labels: List[str] = None) -> Dict[str, Any]:
        """
        Runs evaluation on a test benchmark set and returns comprehensive scikit-learn metrics.
        """
        if test_texts is None or true_labels is None:
            test_texts, true_labels = self._get_benchmark_eval_dataset()

        predicted_labels = []
        confidences = []

        for text in test_texts:
            res = self.classifier.classify(text)
            predicted_labels.append(res["predicted_category"])
            confidences.append(res["confidence"])

        # Compute Scikit-learn metrics
        accuracy = float(accuracy_score(true_labels, predicted_labels))
        prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted", zero_division=0)
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="macro", zero_division=0)

        # Confusion Matrix
        labels_order = self.categories
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)

        # Per-class metrics
        p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels_order, zero_division=0
        )

        per_class_metrics = {}
        for idx, cat in enumerate(labels_order):
            per_class_metrics[cat] = {
                "precision": round(float(p_class[idx]), 3),
                "recall": round(float(r_class[idx]), 3),
                "f1_score": round(float(f1_class[idx]), 3),
                "support": int(support_class[idx])
            }

        return {
            "total_samples": len(true_labels),
            "accuracy": round(accuracy, 4),
            "weighted_precision": round(float(prec), 4),
            "weighted_recall": round(float(rec), 4),
            "weighted_f1": round(float(f1), 4),
            "macro_f1": round(float(macro_f1), 4),
            "average_confidence": round(float(np.mean(confidences)), 4),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "labels": labels_order,
            "classification_report_str": classification_report(true_labels, predicted_labels, labels=labels_order, zero_division=0)
        }

    def _get_benchmark_eval_dataset(self) -> Tuple[List[str], List[str]]:
        """
        Constructs a holdout benchmark dataset distinct from initial training samples
        to evaluate generalizability.
        """
        texts = [
            # Invoices
            "INVOICE Bill To: Vanguard Logistics Inc. Invoice #: INV-1004. Date: 08/12/2024. Description: Fleet telematics subscription $3,200. Subtotal: $3,200. Tax: $320. Total: $3,520. Payment terms: 15 days.",
            "FINAL TAX INVOICE From: CloudHost GmbH. Customer: FinTech Solutions. Invoice Date: 2024-04-11. Total Due: $1,450.00. Subtotal: $1,250.00. VAT 16%: $200.00.",
            "STATEMENT AND INVOICE # 99281. Date: May 4, 2024. Items: Web Development 50 hrs @ $100 = $5,000. Balance Due: $5,000.00. Payment due upon receipt.",
            
            # Resumes
            "Elena Rostova Principal Data Architect. Email: elena.rostova@clouddata.io | Phone: (650) 555-0199. Skills: Apache Spark, Snowflake, Python, PyTorch, Docker, Kubernetes, AWS. Experience: 6 years at Palantir as Lead Data Engineer. Education: M.S. Data Science, UC Berkeley.",
            "Marcus Brody Senior Frontend Developer. Contact: marcus.b@devmail.org. Portfolio: marcusbrody.dev. Technologies: TypeScript, React, Next.js, Redux, CSS3, Tailwind, Jest. Experience: 4 years as UI Engineer at Shopify. Education: B.S. in Computer Science.",
            "CURRICULUM VITAE Dr. Aris Thorne. Email: a.thorne@genomics.org. Research Experience: Postdoctoral fellow at Broad Institute. Skills: Python, R, Deep Learning, Genomics, BioNLP, Scikit-learn. Education: Ph.D. in Computational Biology, Harvard University.",

            # Contracts
            "MUTUAL NON-DISCLOSURE AGREEMENT This NDA is entered into between Zenith Systems ('Disclosing Party') and Apex Ventures ('Receiving Party'). Term: 2 years. Governing Law: State of California. In witness whereof.",
            "SOFTWARE AS A SERVICE (SaaS) AGREEMENT Licensor hereby grants to Customer a subscription to access the Analytics Platform. Limitation of Liability: Capped at fees paid in prior 12 months. Jurisdiction: State of New York.",
            "INDEPENDENT CONTRACTOR AGREEMENT Company engages Contractor to perform machine learning engineering services. Intellectual Property: Work product shall constitute work-made-for-hire. Termination: 14 days notice.",

            # Identity Documents
            "PASSPORT United States of America. Type: P Passport No: 778921094. Surname: JOHNSON Given Names: MICHAEL DAVID. Nationality: USA. Date of Birth: 04 FEB 1989. Sex: M. Authority: US Dept of State.",
            "DRIVER LICENSE State of New York. ID: NY-88910249. Exp: 09/14/2029. DOB: 06/25/1993. Name: SARAH LYNN PARKER. Address: 120 Broadway, New York, NY. Class: D.",
            "NATIONAL IDENTITY CARD Republic of France. ID No: FR-9018401. Nom: DUBOIS Prenom: CLAIRE. Date de Naissance: 12.11.1991. Nationalite: FRANCAISE.",

            # Financial Reports
            "FOURTH QUARTER AND FULL YEAR FINANCIAL STATEMENT Total Revenue was $84.2 Billion, up 12% YoY. Operating Income reached $22.1 Billion. Cash Flow from Operations was $19.4 Billion. Diluted Earnings Per Share of $2.10.",
            "ANNUAL FINANCIAL REPORT FY2023. Gross Profit amounted to $450 Million with an EBITDA margin of 28.4%. Net Sales increased 9.5% across European markets. Long-term debt decreased by $40M.",
            "EARNINGS SUMMARY Q2 2024. Operating Profit: $3.2B. Revenue: $14.5B. Management highlights strong subscriber growth and recurring revenue.",

            # Research Reports
            "TRANSFORMER ATTENTION MECHANISMS IN DOCUMENT RETRIEVAL Abstract: Self-attention has emerged as the state-of-the-art paradigm in document classification. In this work, we propose a multi-scale cross-attention module. Experimental Results demonstrate an F1-score of 94.6%. Methodology and Evaluation.",
            "TECHNICAL REPORT: ZERO-SHOT OCR ADAPTATION Abstract: Degradation in scanned documents severely degrades OCR accuracy. We benchmark convolutional denoising autoencoders on historic archives. Section 2: Related Work. Section 3: Empirical Findings.",
            "RESEARCH PAPER: EVALUATION OF DEEP LEARNING ARCHITECTURES FOR TEXT EXTRACTION Abstract: We present a comprehensive benchmarking of deep neural networks across 5,000 multi-page documents. References and Bibliography.",

            # Receipts
            "TARGET STORE # 2049 Date: 05/18/2024 11:22 AM. Order # 88219. Organic Milk $4.29 Cereal $5.49 Coffee Beans $11.99. Subtotal: $21.77. Tax: $1.74. Total: $23.51. Paid with VISA ending in 4019. Thank you!",
            "SHELL OIL Station 8840 Pump 02 Unleaded Fuel 10.50 Gal @ $3.799 = $39.89. Car Wash $8.00. Total Due: $47.89. Payment: DEBIT CARD APPROVED. Merchant # 99420.",
            "CAFE NERO Date: 04/10/2024. 1x Cappuccino $4.50 1x Croissant $3.75. Subtotal: $8.25. Tax: $0.74. Total: $8.99. Cash Tendered: $10.00 Change: $1.01."
        ]

        labels = [
            "Invoice", "Invoice", "Invoice",
            "Resume", "Resume", "Resume",
            "Contract", "Contract", "Contract",
            "Identity_Document", "Identity_Document", "Identity_Document",
            "Financial_Report", "Financial_Report", "Financial_Report",
            "Research_Report", "Research_Report", "Research_Report",
            "Receipt", "Receipt", "Receipt"
        ]

        return texts, labels
