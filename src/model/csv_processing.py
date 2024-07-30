from dotenv import load_dotenv
import os
import csv
from langchain_community.document_loaders.csv_loader import CSVLoader
from utils_processing import load_documents, split_text, save_to_chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Charger les variables d'environnement
load_dotenv()

# Configurations
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))  # Ajustez cette valeur selon votre système
DATA_PATH_CSV = '../data/output_data.csv'
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
HF_TOKEN = os.getenv('API_TOKEN')

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"Document(page_content={self.page_content[:20]}, metadata={self.metadata})"

def convert_rows_to_documents(rows, header):
    """
    Convertit les lignes CSV en documents.

    Args:
        rows (list of list): Les lignes du fichier CSV.
        header (list): L'en-tête du fichier CSV.

    Returns:
        list of Document: Une liste de documents avec le contenu et les métadonnées des lignes.
    """
    documents = []
    for row in rows:
        content = ' | '.join(f"{header[i]}: {row[i]}" for i in range(len(header)))
        metadata = {}  # Vous pouvez ajouter des métadonnées si nécessaire
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents

def generate_data_store(start_chunk):
    try:
        # Définir la fonction d'embedding
        embedding_function = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="intfloat/multilingual-e5-large")
        
        with open(DATA_PATH_CSV, 'r', encoding='UTF-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Lire l'en-tête
            chunk_size = 1000
            chunk = []
            chunk_index = 0

            for row in reader:
                if chunk_index < start_chunk:
                    # Sauter les lignes jusqu'au bloc de départ
                    if len(chunk) >= chunk_size:
                        chunk = []  # Réinitialiser le bloc après chaque bloc complet
                        chunk_index += 1
                    chunk.append(row)
                    continue

                chunk.append(row)
                if len(chunk) >= chunk_size:
                    # Convertir le bloc en documents et sauvegarder
                    documents = convert_rows_to_documents(chunk, header)
                    save_to_chroma(documents, CHROMA_PATH, COLLECTION_CSV, embedding_function)
                    chunk = []  # Réinitialiser le bloc
                    chunk_index += 1
                    print(f"Processed chunk {chunk_index}")

            # Traiter le reste des lignes qui ne font pas partie d'un bloc complet
            if chunk:
                documents = convert_rows_to_documents(chunk, header)
                save_to_chroma(documents, CHROMA_PATH, COLLECTION_CSV, embedding_function)
                print(f"Processed final chunk {chunk_index + 1}")

    except Exception as e:
        print(f"An error occurred in generate_data_store: {e}")

if __name__ == "__main__":
    generate_data_store(start_chunk=20)
