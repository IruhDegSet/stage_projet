from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from utils_processing import load_documents, split_text, save_to_chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))  # Ajustez cette valeur selon votre système
DATA_PATH_TXT = os.path.abspath(f"../{os.getenv('DATA_PATH_TXT')}")
COLLECTION_TXT = os.getenv('COLLECTION_TXT')
HF_TOKEN = os.getenv('API_TOKEN')
def generate_data_store():
    try:
        loader = TextLoader(file_path=DATA_PATH_TXT, encoding='UTF-8')
        documents = load_documents(loader)
        if not documents:
            return

        chunks = split_text(documents)
        if not chunks:
            return
              # Assurez-vous de définir embedding_function avant de l'utiliser
        embedding_function = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="intfloat/multilingual-e5-large")
        save_to_chroma(chunks, CHROMA_PATH, COLLECTION_TXT, BATCH_SIZE, embedding_function)
    except Exception as e:
        print(f"An error occurred in generate_data_store: {e}")

if __name__ == "__main__":
    generate_data_store()
