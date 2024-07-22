from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import pandas as pd
import os
import shutil
import pickle
import time

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data_base.csv"
BATCH_SIZE = 10000  # Change this to a value that works for your system
DOCUMENTS_PATH = "stored_documents.pkl"  # Path to store the list of documents

def generate_data_store():
    try:
        documents = load_documents()
        if not documents:
            print("No documents were loaded.")
            return
        print(f"Loaded {len(documents)} documents.")
        chunks = split_text(documents)
        if not chunks:
            print("No chunks were created.")
            return
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
        return []

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
        return []

def chunk_documents(documents, batch_size):
    """Divide documents into smaller batches."""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def store_documents(documents):
    """Store the documents in a file."""
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump(documents, f)

def save_to_chroma(chunks: list[Document]):
    try:
        # Clear out the database first.
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # Create a valid embedding object for Hugging Face
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create a new Chroma instance
        print("Creating Chroma instance...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

        # Add documents in batches
        for batch_index, batch in enumerate(chunk_documents(chunks, BATCH_SIZE)):
            print(f"Processing batch {batch_index + 1} of {len(chunks)//BATCH_SIZE + 1}...")
            print(f"Adding batch of {len(batch)} chunks to Chroma...")
            start_time = time.time()
            try:
                db.add_documents(documents=batch)
                print(f"Batch {batch_index + 1} added in {time.time() - start_time:.2f} seconds.")
            except Exception as e:
                print(f"An error occurred while adding batch {batch_index + 1}: {e}")
                break  # Stop processing if an error occurs

        db.persist()
        print(f"Saved all chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"An error occurred in save_to_chroma: {e}")

if __name__ == "__main__":
    generate_data_store()
