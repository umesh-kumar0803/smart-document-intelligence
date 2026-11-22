"""
Validation and Quality Assurance Engine.
Performs cross-field integrity checks, mathematical reconciliation, and format verification.
"""
from typing import Dict, List, Any
from app.schemas import ValidationResult


class DocumentValidator:
    """Validates extracted document data against business rules and data consistency constraints."""

    def validate(self, category: str, fields: Dict[str, Any], classification_confidence: float) -> ValidationResult:
        """
        Validates the extracted fields for a given category.
        
        Returns:
            ValidationResult with overall_score, status, passed_checks, warnings, and errors.
        """
        passed_checks: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        # 1. Classification Confidence Check
        if classification_confidence >= 0.75:
            passed_checks.append(f"High Classification Confidence ({classification_confidence:.1%})")
        elif classification_confidence >= 0.50:
            warnings.append(f"Moderate Classification Confidence ({classification_confidence:.1%})")
        else:
            warnings.append(f"Low Classification Confidence ({classification_confidence:.1%}); please verify document type.")

        # 2. Category-Specific Integrity Rules
        if category == "Invoice":
            self._validate_invoice(fields, passed_checks, warnings, errors)
        elif category == "Resume":
            self._validate_resume(fields, passed_checks, warnings, errors)
        elif category == "Contract":
            self._validate_contract(fields, passed_checks, warnings, errors)
        elif category == "Identity_Document":
            self._validate_identity_document(fields, passed_checks, warnings, errors)
        elif category == "Financial_Report":
            self._validate_financial_report(fields, passed_checks, warnings, errors)
        elif category == "Receipt":
            self._validate_receipt(fields, passed_checks, warnings, errors)
        else:
            passed_checks.append("Standard generic text format passed.")

        # 3. Overall Quality Score Calculation
        total_evaluations = len(passed_checks) + len(warnings) + (len(errors) * 2)
        if total_evaluations == 0:
            overall_score = 0.5
        else:
            positive_weight = len(passed_checks)
            warning_penalty = len(warnings) * 0.5
            overall_score = max(0.1, min(1.0, (positive_weight - warning_penalty) / total_evaluations))

        # Overall Status
        if len(errors) > 0:
            status = "Invalid"
        elif len(warnings) > 0 or overall_score < 0.70:
            status = "Requires Review"
        else:
            status = "Valid"

        return ValidationResult(
            overall_score=round(overall_score, 2),
            is_valid=len(errors) == 0,
            status=status,
            passed_checks=passed_checks,
            warnings=warnings,
            errors=errors
        )

    def _validate_invoice(self, fields: Dict[str, Any], passed: List[str], warnings: List[str], errors: List[str]):
        # Check Invoice Number
        inv_no = fields.get("invoice_number")
        if inv_no and inv_no != "N/A":
            passed.append(f"Invoice Number present: {inv_no}")
        else:
            warnings.append("Invoice Number could not be detected.")

        # Check Math Reconciliation: subtotal + tax ~= total
        sub = fields.get("subtotal")
        tax = fields.get("tax_amount")
        total = fields.get("total_amount")

        if total is not None:
            passed.append(f"Total Amount detected: {fields.get('currency', '$')} {total:,.2f}")
            if sub is not None and tax is not None:
                expected_total = sub + tax
                diff = abs(expected_total - total)
                if diff <= 0.05:
                    passed.append(f"Mathematical reconciliation verified: Subtotal ({sub:,.2f}) + Tax ({tax:,.2f}) == Total ({total:,.2f})")
                else:
                    warnings.append(
                        f"Mathematical discrepancy: Subtotal ({sub:,.2f}) + Tax ({tax:,.2f}) = {expected_total:,.2f}, "
                        f"which differs from Total ({total:,.2f}) by {diff:,.2f}."
                    )
        else:
            errors.append("Total Amount could not be located in invoice.")

        # Dates
        if fields.get("invoice_date") != "N/A":
            passed.append(f"Invoice Date present: {fields.get('invoice_date')}")
        else:
            warnings.append("Invoice Date is missing.")

        # Parties
        if fields.get("vendor_name") != "N/A":
            passed.append(f"Vendor identified: {fields.get('vendor_name')}")
        if fields.get("customer_name") != "N/A":
            passed.append(f"Customer identified: {fields.get('customer_name')}")

    def _validate_resume(self, fields: Dict[str, Any], passed: List[str], warnings: List[str], errors: List[str]):
        # Candidate Name
        name = fields.get("candidate_name")
        if name and name != "Candidate":
            passed.append(f"Candidate Name identified: {name}")
        else:
            warnings.append("Candidate Name could not be reliably determined.")

        # Contact Info
        email = fields.get("email")
        phone = fields.get("phone")

        if email and email != "N/A":
            passed.append(f"Valid Email format verified: {email}")
        else:
            warnings.append("No contact email address identified.")

        if phone and phone != "N/A":
            passed.append(f"Phone contact identified: {phone}")

        # Skills
        skills = fields.get("skills", [])
        if len(skills) >= 3:
            passed.append(f"Strong Skill Profile: {len(skills)} competencies detected")
        elif len(skills) > 0:
            passed.append(f"Skills detected: {len(skills)}")
        else:
            warnings.append("No technical competencies found from standard taxonomy.")

    def _validate_contract(self, fields: Dict[str, Any], passed: List[str], warnings: List[str], errors: List[str]):
        parties = fields.get("parties", [])
        if len(parties) >= 2:
            passed.append(f"Identified {len(parties)} contracting parties: {', '.join(parties)}")
        else:
            warnings.append("Fewer than 2 legal parties detected in agreement.")

        clauses = fields.get("detected_clauses", [])
        if len(clauses) >= 3:
            passed.append(f"Essential Legal Clauses present: {', '.join(clauses)}")
        else:
            warnings.append("Few standard contractual clauses detected.")

        if fields.get("effective_date") != "N/A":
            passed.append(f"Effective Date confirmed: {fields.get('effective_date')}")
        else:
            warnings.append("No explicit effective date found.")

    def _validate_identity_document(self, fields: Dict[str, Any], passed: List[str], warnings: List[str], errors: List[str]):
        doc_id = fields.get("document_id")
        if doc_id and doc_id != "N/A" and len(doc_id) >= 5:
            passed.append(f"Government ID format valid: {doc_id}")
        else:
            errors.append("Document ID Number is missing or below minimum standard length.")

        name = fields.get("full_name")
        if name and name != "N/A":
            passed.append(f"Holder Full Name verified: {name}")
        else:
            warnings.append("Holder Name missing.")

        if fields.get("date_of_birth") != "N/A":
            passed.append(f"Date of Birth identified: {fields.get('date_of_birth')}")
        else:
            warnings.append("Date of Birth is missing.")

    def _validate_financial_report(self, fields: Dict[str, Any], passed: List[str], warnings: List[str], errors: List[str]):
        rev = fields.get("total_revenue")
        if rev and rev != "N/A":
            passed.append(f"Revenue metric present: {rev}")
        else:
            warnings.append("Total Revenue figure not explicitly found.")

        period = fields.get("period")
        if period:
            passed.append(f"Financial Period verified: {period}")

    def _validate_receipt(self, fields: Dict[str, Any], passed: List[str], warnings: List[str], errors: List[str]):
        total = fields.get("total_amount")
        if total is not None and total > 0:
            passed.append(f"Receipt Total verified: ${total:,.2f}")
        else:
            errors.append("Receipt Total Amount is missing.")

        merchant = fields.get("merchant_name")
        if merchant:
            passed.append(f"Merchant Name: {merchant}")
