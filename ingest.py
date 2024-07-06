from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Define the new Chroma client settings here
CHROMA_SETTINGS = {
    "chroma_api_impl": "rest",
    "api_base_url": "http://localhost:8000"  # Update this with your Chroma server URL if needed
}

persist_directory = "db"

def main():
    loader = None  # Initialisation de loader en dehors de la boucle
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    if loader is not None:
        documents = loader.load()
        print("splitting into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        # Création des embeddings ici
        print("Loading sentence transformers model")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Création du vecteur store ici
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(
            texts, 
            embeddings, 
            persist_directory=persist_directory
        )
        db.persist()
        db = None

        print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    else:
        print("No PDF files found to load.")

if __name__ == "__main__":
    main()
