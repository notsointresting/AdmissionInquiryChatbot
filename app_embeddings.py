from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
from news_scrapper import main # For getting latest news so everytime i run app_embeddings it will get the latest news also.

main()

# Setting path for input data files
DATA_PATH = 'data/'

# Path for vectorstore to store text embeddings made from the data
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Google Gemini API key setup
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv("API_KEY"))

def create_vector_db():
    # Load the PDF documents from 'data/' and all its subdirectories
    print(f"Loading documents from: {DATA_PATH} and its subdirectories")
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader, 
        recursive=True, 
        show_progress=True, # Show a progress bar while loading
        use_multithreading=True # Use multiple threads to load files faster
    )
    documents = loader.load()
    if not documents:
        print("No PDF documents found. Exiting.")
        return
    print(f"Number of documents loaded: {len(documents)}")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    print(f"Number of text chunks created: {len(texts)}")

    # Check the structure of texts
    if texts and hasattr(texts[0], 'page_content'):
        print(f"Example text chunk: {texts[0].page_content}")
    else:
        raise TypeError("Texts are not in the expected format of a list of Document objects")

    # Using Google Generative AI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Split texts into batches and add to vector store incrementally
    batch_size = 50
    total_texts = len(texts)


    # Initialize FAISS vector store
    db = None

    for i in range(0, total_texts, batch_size):
        batch = texts[i:i + batch_size]
        batch_content = [doc.page_content for doc in batch]
        print(f"Processing batch {i // batch_size + 1} of size: {len(batch)}")
        try:
            batch_embeddings = embeddings.embed_documents(batch_content)
            if db is None:
                # Create the FAISS vector store with the first batch
                db = FAISS.from_texts(batch_content, embeddings)
            else:
                # Create dummy metadata
                metadatas = [{} for _ in batch]
                # Add the embeddings to the existing FAISS vector store
                db.add_texts(batch_content, metadatas=metadatas)
            print(f"Successfully processed batch {i // batch_size + 1}")
            # Reduced sleep time, but still respectful of potential rate limits
            # Google's default is often 60 requests per minute for embeddings.
            # If batch_size is 50, one call per batch. This sleep might be conservative.
            print("Sleeping for 10 seconds before next batch...")
            time.sleep(10) 
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            raise

    # Saving the embeddings in the vector store
    if db is not None:
        db.save_local(DB_FAISS_PATH)
        print("Successfully made and saved text embeddings!")
    else:
        print("No documents were processed.")

if __name__ == "__main__":
    create_vector_db()
