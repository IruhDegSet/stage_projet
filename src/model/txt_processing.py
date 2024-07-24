from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from utils_processing import load_documents, split_text, save_to_chroma

load_dotenv()


CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))  # Ajustez cette valeur selon votre système
DATA_PATH = os.path.abspath(f"../{os.getenv('DATA_PATH_TXT')}")

def generate_data_store():
    try:
        loader = TextLoader(file_path=DATA_PATH, encoding='UTF-8')
        documents = load_documents(loader)
        if not documents:
            return

        chunks = split_text(documents)
        if not chunks:
            return

        save_to_chroma(chunks, CHROMA_PATH, BATCH_SIZE)
    except Exception as e:
        print(f"An error occurred in generate_data_store: {e}")

if __name__ == "__main__":
    generate_data_store()
