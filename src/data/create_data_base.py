from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import pandas as pd
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data_base.csv"
BATCH_SIZE = 5000  # Change this to a value that works for your system

def generate_data_store():
    try:
        documents = load_documents()
        print(f"Loaded {len(documents)} documents.")
        chunks = split_text(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        save_to_chroma(chunks)
    except Exception as e:
        print(f"An error occurred in generate_data_store: {e}")

def load_documents():
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"CSV columns: {df.columns}")
        documents = []
        for index, row in df.sample(frac=0.05).iterrows():  # Sample 5% of the data randomly
            content = row['description']
            metadata = {
                "part": row['part'],
                "fournisseur": row['fournisseur'],
                "marque": row['marque'],
                "prix": row['prix'],
                "quantity": row['quantity']
            }
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        return documents
    except Exception as e:
        print(f"An error occurred in load_documents: {e}")

def split_text(documents: list[Document]):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        print(f"An error occurred in split_text: {e}")

def chunk_documents(documents, batch_size):
    """Divide documents into smaller batches."""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def save_to_chroma(chunks: list[Document]):
    try:
        # Clear out the database first.
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # Create a valid embedding object for Hugging Face
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create a new Chroma instance
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

        # Add documents in batches
        for batch in chunk_documents(chunks, BATCH_SIZE):
            db.add_documents(
                documents=batch
            )
            print(f"Added a batch of {len(batch)} chunks to Chroma.")

        db.persist()
        print(f"Saved all chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"An error occurred in save_to_chroma: {e}")

if __name__ == "__main__":
    generate_data_store()
