from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
import shutil
import time
import traceback
from dotenv import load_dotenv

load_dotenv()  # Charge les variables d'environnement depuis un fichier .env

HF_TOKEN = os.getenv('API_TOKEN')

def load_documents(loader):
    try:
        documents = loader.load()
        if not documents:
            print("No documents were loaded.")
            return []

        print(f"Loaded {len(documents)} documents.")
        return documents
    except FileNotFoundError:
        print("Error: The file does not exist.")
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
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def save_to_chroma(chunks: list[Document], CHROMA_PATH, BATCH_SIZE):
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        embedding = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="intfloat/multilingual-e5-large")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

        for batch_index, batch in enumerate(chunk_documents(chunks, BATCH_SIZE)):
            print(f"Processing batch {batch_index + 1} of {len(chunks)//BATCH_SIZE + 1}...")
            start_time = time.time()
            try:
                db.add_documents(documents=batch)
                print(f"Batch {batch_index + 1} added in {time.time() - start_time:.2f} seconds.")
            except Exception as e:
                print(f"An error occurred while adding batch {batch_index + 1}: {e}")
                break

        db.persist()
        print(f"Saved all chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"An error occurred in save_to_chroma: {e}")

