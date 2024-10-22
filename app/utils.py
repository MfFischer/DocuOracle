import PyPDF2
import docx
import pandas as pd

def parse_pdf(file_path):
    """
    Extracts text from a PDF file.
    :param file_path: Path to the PDF file
    :return: Extracted text from the PDF
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text += page.extractText()
            return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def parse_word(file_path):
    """
    Extracts text from a Word (docx) file.
    :param file_path: Path to the Word file
    :return: Extracted text from the Word document
    """
    try:
        doc = docx.Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading Word file: {e}")
        return None

def parse_excel(file_path):
    """
    Reads an Excel file and returns it as a Pandas DataFrame.
    :param file_path: Path to the Excel file
    :return: DataFrame containing the Excel data
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def save_uploaded_file(uploaded_file, upload_folder='documents'):
    """
    Saves an uploaded file to the specified directory.
    :param uploaded_file: File object from the form
    :param upload_folder: Directory where the file will be saved
    :return: Full file path of the saved file
    """
    try:
        filename = uploaded_file.filename
        file_path = f"{upload_folder}/{filename}"
        uploaded_file.save(file_path)
        return file_path
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return None
