from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import os
import shutil
import pickle
import time
from langchain_community.document_loaders.csv_loader import CSVLoader
import traceback
from langchain_community.document_loaders import DirectoryLoader
# Charger les variables d'environnement
load_dotenv()

CHROMA_PATH = "../data/chroma"
DATA_PATH = "../data/sample_db.txt"
BATCH_SIZE = 10000  # Ajustez cette valeur selon votre système


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
        # Vérifiez si le fichier existe
        if not os.path.exists(DATA_PATH):
            print(f"Error: The file {DATA_PATH} does not exist.")
            return []

        # Utiliser CSVLoader pour lire le fichier CSV
        print(f"Attempting to load CSV file from {DATA_PATH}")
        loader = CSVLoader(DATA_PATH)
        documents = loader.load()  # Attendre une liste d'objets Document

        # Vérifiez si les documents sont chargés correctement
        if not documents:
            print("No documents were loaded from the CSV file.")
            return []

        # Optionnel : Afficher les détails des documents pour le débogage
        print(f"Loaded {len(documents)} documents.")
        return documents

    except FileNotFoundError:
        print(f"Error: The file {DATA_PATH} does not exist.")
    except Exception as e:
        print(f"An error occurred in load_documents: {e}")
        print(traceback.format_exc())
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

def chunk_documents(documents, batch_size):
    """Diviser les documents en petits lots."""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def save_to_chroma(chunks: list[Document]):
    try:
        # Nettoyer la base de données d'abord
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # Créer un objet d'embedding valide pour Hugging Face
        embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

        # Créer une nouvelle instance Chroma
        print("Creating Chroma instance...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

        # Ajouter les documents par lots
        for batch_index, batch in enumerate(chunk_documents(chunks, BATCH_SIZE)):
            print(f"Processing batch {batch_index + 1} of {len(chunks)//BATCH_SIZE + 1}...")
            print(f"Adding batch of {len(batch)} chunks to Chroma...")
            start_time = time.time()
            try:
                db.add_documents(documents=batch)
                print(f"Batch {batch_index + 1} added in {time.time() - start_time:.2f} seconds.")
            except Exception as e:
                print(f"An error occurred while adding batch {batch_index + 1}: {e}")
                break  # Arrêter le traitement si une erreur survient

        db.persist()
        print(f"Saved all chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"An error occurred in save_to_chroma: {e}")

if __name__ == "__main__":
    generate_data_store()
