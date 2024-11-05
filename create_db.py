import os
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

# Set up your Google Gemini API key
GEMINI_API_KEY = "AIzaSyDFQrUxPXyeVGU66oxymNMeK9IZy_Z272U"

# Load all PDFs from the specified folder
pdf_folder_path = "./Data/"
all_pdf_paths = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))

# Load each PDF document and split text
documents = []
for pdf_path in all_pdf_paths:
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents.extend(text_splitter.split_documents(pdf_docs))

print(f"Total loaded document chunks: {len(documents)}")

# Set up embeddings with Google Gemini API
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# Create embeddings for documents
document_embeddings = embeddings.embed_documents(documents)

# Create Chroma vector database from documents and their embeddings
vector_db = Chroma.from_embeddings(document_embeddings, documents)

# Optionally, save the database to disk for later use
vector_db.save("my_chroma_db")  # Specify the filename to save

print("Database has been created and saved.")