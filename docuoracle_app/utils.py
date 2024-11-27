import PyPDF2
import docx
import pandas as pd
from typing import Optional, Union
from PyPDF2 import PdfReader
from docx import Document
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_pdf(filepath):
    """Parse PDF file and return text content."""
    try:
        text = []
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    except Exception as e:
        logger.error(f"Error parsing PDF file: {str(e)}")
        return None


def parse_word(filepath):
    """Parse Word document and return text content."""
    try:
        doc = docx.Document(filepath)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error parsing Word file: {str(e)}")
        return None


def parse_excel(filepath):
    """Parse Excel or CSV file and return pandas DataFrame."""
    try:
        # Get file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext in ['.xlsx', '.xls']:
            # For Excel files, specify engine explicitly
            engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
            return pd.read_excel(filepath, engine=engine)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return None
    except Exception as e:
        logger.error(f"Error parsing Excel/CSV file: {str(e)}")
        return None

def parse_document(filepath):
    """Parse document based on file type."""
    try:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None

        # Get file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext in ['.xlsx', '.xls', '.csv']:
            return parse_excel(filepath)
        elif ext == '.pdf':
            return parse_pdf(filepath)
        elif ext == '.docx':
            return parse_word(filepath)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return None
    except Exception as e:
        logger.error(f"Error parsing document: {str(e)}")
        return None

def get_document_columns(filepath):
    """Get columns from data file."""
    try:
        df = parse_excel(filepath)
        if df is not None:
            return {
                'success': True,
                'columns': {
                    'numeric': df.select_dtypes(include=['number']).columns.tolist(),
                    'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
                    'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                    'all': df.columns.tolist()
                }
            }
        return {
            'success': False,
            'error': 'Could not parse file'
        }
    except Exception as e:
        logger.error(f"Error getting columns: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
