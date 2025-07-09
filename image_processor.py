import os
import pytesseract
from PIL import Image
from fpdf import FPDF

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
        raise
    except Exception as e:
        print(f"Error processing image {image_path} with Tesseract: {e}")
        return None

def create_pdf_from_text(text_content, original_image_filename, output_pdf_path):
    """Creates a PDF document from the extracted text."""
    pdf = FPDF()
    pdf.add_page()
    
    active_font = 'Arial'
    try:
        pdf.set_font(active_font, '', 12)
    except RuntimeError as e:
        print(f"Warning: Could not add DejaVu font ({e}), using Arial. Special characters might not render correctly.")
        pdf.set_font('Arial', '', 12)
    
    pdf.set_title(f"Extracted Text from {original_image_filename}")
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
    if not os.path.isdir(input_dir):
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
            
            if text is not None and text.strip():
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
    dummy_input_dir = "data/images_to_process"
    if not os.path.exists(dummy_input_dir):
        os.makedirs(dummy_input_dir)
        print(f"Created dummy input directory: {dummy_input_dir}")
        print(f"Please add some images (e.g., PNG, JPG) to {dummy_input_dir} to test the processor.")
    
    print("Starting image processing...")
    try:
        process_images_in_directory(input_dir=dummy_input_dir)
    except pytesseract.TesseractNotFoundError:
        print("Image processing halted due to Tesseract OCR not being found or configured correctly.")
        print("Please ensure Tesseract is installed and in your system PATH, or configure pytesseract.tesseract_cmd.")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
    print("Image processing script finished.")
