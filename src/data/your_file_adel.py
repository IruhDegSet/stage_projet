import os
import shutil
import requests
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import Chroma
from IPython.display import display, Markdown
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Chargement des variables d'environnement
_ = load_dotenv(find_dotenv())

# Définir le chemin pour le stockage de la base de données Chroma
CHROMA_PATH = 'chroma_test'

# Chargement des données CSV
file = 'data_base.csv'
df = pd.read_csv(file, encoding='utf-8')
data = df.sample(10, random_state=42)
data.to_csv('sample_db.csv', index=False)

# Conversion des données en texte
columns = data.columns
doc = '\n'.join([
    ' '.join([f"{col}: {str(row[col]).strip()}" for col in columns])
    for idx, row in data.iterrows()
])

# Écriture du texte dans un fichier avec encodage UTF-8
with open('sample_db.txt', 'w', encoding='utf-8') as f:
    f.write(doc)

# Analyse des longueurs de lignes
lengths = [len(line) for line in doc.split('\n')]
print("Mean length:", np.mean(lengths))
print("Min length:", np.min(lengths))
print("Max length:", np.max(lengths))
print("Median length:", np.median(lengths))

# Chargement du fichier texte
file = 'sample_db.txt'
loader = TextLoader(file_path=file, encoding='utf-8')

# Charger les documents à partir du fichier texte
try:
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
except Exception as e:
    print("Error loading documents:", e)

# Définition du splitter de texte
r_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# Diviser les documents en morceaux
try:
    chunks = r_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
except Exception as e:
    print("Error splitting documents:", e)

# Créer un objet d'embedding valide pour Hugging Face
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Création de l'index avec Chroma
print("Creating Chroma index...")
try:
    # Nettoyer la base de données d'abord si elle existe
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Créer un nouvel index Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()
    print("Chroma index created.")
except Exception as e:
    print("Error creating Chroma index:", e)

# Initialisation du modèle ChatGroq
llm_replacement_model = ChatGroq(
    temperature=1,
    model='llama3-8b-8192',
    groq_api_key='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'
)

# Créez un retriever à partir de Chroma
retriever = vectorstore.as_retriever()

# Créez une instance de RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_replacement_model,
    retriever=retriever
)

# Exécution de la requête
query = "afficher tout les produits"
print("Executing query...")
try:
    response = qa_chain.run(query)
    print("Query executed.")
    print("Response:", response)
except Exception as e:
    print("Error executing query:", e)
try:
    print("Loading Chroma index...")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    print("Chroma index loaded.")
    
    # Compter les éléments (documents) dans la base de données
    # Cela suppose que vous pouvez accéder aux documents ou leurs métadonnées
    num_documents = len(vectorstore)  # Utiliser la longueur pour obtenir le nombre de documents
    print(f"Number of elements in Chroma database: {num_documents}")

except Exception as e:
    print("Error loading Chroma index or counting elements:", e)