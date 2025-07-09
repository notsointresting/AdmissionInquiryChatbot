import subprocess
import sys
import os
import datetime

def run_script(script_name):
    """Runs a Python script and prints its output in real-time."""
    print(f"\n----- Running {script_name} at {datetime.datetime.now()} -----")
    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.exists(script_path):
            script_path = script_name 
            if not os.path.exists(script_path):
                print(f"Error: Script '{script_name}' not found at '{os.path.join(os.path.dirname(__file__), script_name)}' or as '{script_name}'.")
                return -1

        process = subprocess.Popen([sys.executable, script_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   universal_newlines=True)

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            process.stdout.close()
        
        process.wait()

        if process.returncode == 0:
            print(f"----- {script_name} completed successfully at {datetime.datetime.now()} -----\n")
        else:
            print(f"----- {script_name} FAILED with return code {process.returncode} at {datetime.datetime.now()} -----\n")
        return process.returncode
    except FileNotFoundError:
        print(f"Error: Script '{script_name}' not found. Make sure it's in the correct path.")
        return -1
    except Exception as e:
        print(f"An critical error occurred while trying to run {script_name}: {e}")
        return -1

def main():
    print(f"=== Starting daily data pipeline at {datetime.datetime.now()} ===")
    
    print("Note: `image_processor.py` requires Tesseract OCR to be installed and accessible.")
    
    images_input_dir = "data/images_to_process"
    if not os.path.exists(images_input_dir):
        try:
            os.makedirs(images_input_dir)
            print(f"Created directory for new images: {images_input_dir}")
        except OSError as e:
            print(f"Error creating directory {images_input_dir}: {e}. Image processing might fail if it relies on this.")
    
    print("\nStep 1: Scraping latest news...")
    news_scrapper_status = run_script("news_scrapper.py")
    if news_scrapper_status != 0:
        print("Warning: News scrapping may have failed or had issues. Check logs above.")

    print("\nStep 2: Processing images for text extraction...")
    image_processor_status = run_script("image_processor.py")
    if image_processor_status != 0:
        print("Warning: Image processing may have failed or had issues (e.g., Tesseract not found). Check logs above.")

    print("\nStep 3: Generating and updating text embeddings...")
    embedding_status = run_script("app_embeddings.py")
    if embedding_status != 0:
        print("CRITICAL: Embedding generation failed. The vector database may not be up-to-date.")
    
    print(f"\n=== Daily data pipeline finished at {datetime.datetime.now()} ===")
    print("To automate this pipeline, schedule this script (run_daily_data_pipeline.py) using cron (Linux/macOS) or Task Scheduler (Windows).")

if __name__ == "__main__":
    main()
