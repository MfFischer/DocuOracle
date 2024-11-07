import PyPDF2
import docx
import pandas as pd
from typing import Optional, Union


def parse_document(filepath: str) -> Optional[Union[str, pd.DataFrame]]:
    """Parse document based on file type."""
    try:
        if filepath.endswith('.pdf'):
            return parse_pdf(filepath)
        elif filepath.endswith('.docx'):
            return parse_word(filepath)
        elif filepath.endswith(('.xlsx', '.xls', '.csv')):
            return parse_excel(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    except Exception as e:
        print(f"Error parsing document: {e}")
        return None


def parse_pdf(filepath: str) -> Optional[str]:
    """Parse PDF file and return its text content."""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None


def parse_word(filepath: str) -> Optional[str]:
    """Parse Word document and return its text content."""
    try:
        doc = docx.Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error parsing Word document: {e}")
        return None


def parse_excel(filepath: str) -> Optional[pd.DataFrame]:
    """Parse Excel file and return pandas DataFrame."""
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error parsing Excel file: {e}")
        return None
