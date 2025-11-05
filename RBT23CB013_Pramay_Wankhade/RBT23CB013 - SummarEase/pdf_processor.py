"""
PDF processing utilities
"""
import PyPDF2

def extract_text_from_pdf(pdf_file):
    """
    Extract text from PDF file with page markers
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        str: Extracted text with page numbers
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def get_pdf_info(pdf_file):
    """
    Get information about the PDF file
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        dict: PDF metadata
    """
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return {
            "page_count": len(pdf_reader.pages),
            "metadata": pdf_reader.metadata if hasattr(pdf_reader, 'metadata') else {}
        }
    except Exception as e:
        return {"page_count": 0, "metadata": {}, "error": str(e)}