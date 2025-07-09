import os
import pytesseract
from PIL import Image
from fpdf import FPDF # Using FPDF from fpdf2 library

# It's good practice to specify the Tesseract command path if it's not in PATH
# For example:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Example for macOS
# Or on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# This should be configured by the user in their environment or a config file.
# For this script to work in the sandbox, Tesseract must be pre-installed and in PATH.

def extract_text_from_image(image_path):
    """Extracts text from a single image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        print(f"Extracted text from {image_path} (first 100 chars): {text[:100].strip()}")
        return text
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except pytesseract.TesseractNotFoundError:
        print("Critical Error: Tesseract is not installed or not in your PATH.")
        print("Please install Tesseract OCR and ensure it's added to your system's PATH.")
        print("See: https://github.com/tesseract-ocr/tesseract#installing-tesseract")
        # Re-raise to stop processing if Tesseract is missing, caught in main.
        raise
    except Exception as e:
        print(f"Error processing image {image_path} with Tesseract: {e}")
        return None

def create_pdf_from_text(text_content, original_image_filename, output_pdf_path):
    """Creates a PDF document from the extracted text."""
    pdf = FPDF()
    pdf.add_page()
    
    active_font = 'Arial' # Default font
    try:
        # Ensure DejaVuSansCondensed.ttf is in a directory FPDF checks (e.g., current dir, or system font dir)
        # Or provide full path to the .ttf file.
        # For the sandbox, this font file would need to be present.
        # pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True) 
        # active_font = 'DejaVu'
        # print("Using DejaVu font for PDF.")
        pdf.set_font(active_font, '', 12) # Use Arial as default for now
    except RuntimeError as e:
        print(f"Warning: Could not add DejaVu font ({e}), using Arial. Special characters might not render correctly.")
        pdf.set_font('Arial', '', 12) # Fallback
    
    pdf.set_title(f"Extracted Text from {original_image_filename}")
    # Encode to latin-1 to prevent FPDF errors with unsupported characters if not using a unicode font
    safe_text_content = text_content.encode('latin-1', 'replace').decode('latin-1')
    safe_original_image_filename = original_image_filename.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 10, f"Extracted text from image: {safe_original_image_filename}\n\n{safe_text_content}")
    
    try:
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        pdf.output(output_pdf_path, 'F')
        print(f"Successfully created PDF: {output_pdf_path}")
    except Exception as e:
        print(f"Error saving PDF to {output_pdf_path}: {e}")

def process_images_in_directory(input_dir="data/images_to_process", output_dir="data/processed_image_pdfs"):
    """
    Processes all images in a given directory, extracts text, and saves each as a PDF.
    """
    if not os.path.isdir(input_dir): # Check if directory exists and is a directory
        print(f"Input directory '{input_dir}' not found or is not a directory. No images to process.")
        return

    os.makedirs(output_dir, exist_ok=True)

    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(input_dir, filename)
            print(f"Processing image: {image_path}...")
            
            text = extract_text_from_image(image_path)
            
            if text is not None and text.strip(): # Ensure text is not None and not just whitespace
                # Sanitize filename for PDF output
                pdf_filename_base = os.path.splitext(filename)[0]
                pdf_filename = f"{pdf_filename_base}_extracted.pdf"
                output_pdf_path = os.path.join(output_dir, pdf_filename)
                create_pdf_from_text(text, filename, output_pdf_path)
                processed_count += 1
            elif text is not None and not text.strip():
                 print(f"No text content extracted from {filename}.")
            else:
                print(f"Skipping PDF creation for {filename} due to extraction error or no text.")
    
    if processed_count == 0:
        print(f"No images were processed or no text was extracted from images in {input_dir}.")
    else:
        print(f"Finished processing images. {processed_count} PDFs created in {output_dir}.")


if __name__ == '__main__':
    # Example usage:
    # This script expects Tesseract OCR to be installed and accessible.
    # Images should be placed in 'data/images_to_process'.
    # Output PDFs will be saved in 'data/processed_image_pdfs'.
    
    # Create dummy image directory for first run if it doesn't exist
    # In a real scenario, this directory would be populated by other means.
    dummy_input_dir = "data/images_to_process"
    if not os.path.exists(dummy_input_dir):
        os.makedirs(dummy_input_dir)
        print(f"Created dummy input directory: {dummy_input_dir}")
        print(f"Please add some images (e.g., PNG, JPG) to {dummy_input_dir} to test the processor.")
    
    print("Starting image processing...")
    try:
        process_images_in_directory(input_dir=dummy_input_dir)
    except pytesseract.TesseractNotFoundError:
        # Error message already printed in extract_text_from_image
        print("Image processing halted due to Tesseract OCR not being found or configured correctly.")
        print("Please ensure Tesseract is installed and in your system PATH, or configure pytesseract.tesseract_cmd.")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
    print("Image processing script finished.")
