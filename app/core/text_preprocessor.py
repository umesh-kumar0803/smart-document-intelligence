"""
Text cleaning, normalization, tokenization, and linguistic preprocessing.
"""
import re
import unicodedata
from typing import Dict, List, Any


class TextPreprocessor:
    """Preprocesses raw OCR/extracted text for NLP models and information extraction."""

    # Common English stopwords for feature representation
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
        "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
        "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
        "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
        "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

    def clean_text(self, text: str) -> str:
        """
        Cleans OCR text:
        - Normalizes unicode characters
        - Removes control characters
        - Standardizes line breaks and whitespace
        - Fixes common hyphenation line-breaks
        """
        if not text:
            return ""

        # Normalize unicode (e.g. smart quotes, non-standard hyphens)
        text = unicodedata.normalize("NFKD", text)

        # Remove control characters except standard whitespace \n, \r, \t
        text = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or unicodedata.category(ch)[0] != "C")

        # Fix OCR hyphenation across lines (e.g. "commer-\ncial" -> "commercial")
        text = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", text)

        # Standardize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace excessive multiple blank lines with double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace multiple horizontal spaces/tabs with single space
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        cleaned = "\n".join(lines).strip()

        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """Simple regex-based word tokenization."""
        return re.findall(r"\b[A-Za-z0-9_$%€£₹.-]+\b", text)

    def extract_clean_tokens_no_stopwords(self, text: str) -> List[str]:
        """Extracts lowercased alphanumeric tokens excluding common stopwords."""
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculates linguistic statistics from the document text."""
        cleaned = self.clean_text(text)
        words = self.tokenize(cleaned)
        lines = [line for line in cleaned.split("\n") if line.strip()]
        
        # Approximate sentence split
        sentences = re.split(r"[.!?]+\s+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]

        char_count = len(cleaned)
        word_count = len(words)
        vocab_size = len(set(w.lower() for w in words))
        
        # Average reading speed: 200 words per minute
        est_reading_time_sec = round((word_count / 200.0) * 60, 1)

        return {
            "character_count": char_count,
            "word_count": word_count,
            "line_count": len(lines),
            "sentence_count": len(sentences),
            "vocabulary_size": vocab_size,
            "lexical_diversity": round(vocab_size / max(1, word_count), 3),
            "estimated_reading_time_sec": est_reading_time_sec,
        }
