__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
try:
    from langchain_community.document_loaders.csv_loader import CSVLoader
except ImportError:
    from langchain.document_loaders.csv_loader import CSVLoader  # Alternative import

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
CHROMA_PATH = "../data/chroma"
BATCH_SIZE = 1000

COLLECTION_CSV = 'csv_collection'
MBD_MODEL = 'intfloat/multilingual-e5-large'

def ask_bot(query: str, k: int = 10):
    persist_directory = CHROMA_PATH
    embedding = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=COLLECTION_CSV)

    # Initiate model
    llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)

    # Build prompt
    template = """tu es un assistant vendeur, tu as acces au context seulement. ne generes pas des infos si elles ne sont pas dans le context il faut repondre seulement si tu as la reponse. accompagne chaque reponse du ref_produit, marque et description du produit tel qu'ils sont dans le context. affiche autant de lignes que les produit trouves dans le context. repond a la question de l'utilisateur en francais. tu est oblige de repondre dans un tableau avec comme colonnes: reference, marque et la description
    {context}
    Question: {question}
    Reponse:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Build chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k': k}),
        return_source_documents=True,
        chain_type='stuff',
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Run chain:
    result = qa_chain.invoke({"query": query})
    st.write(result) 
    return result['result']

def inspect_chroma():
    persist_directory = CHROMA_PATH
    embedding = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=COLLECTION_CSV)

    # Example: Print all document IDs
    st.write("Inspecting Chroma Database...")
    all_docs = vectordb.get_all_documents()
    for doc in all_docs:
        st.write(f"Document ID: {doc.id}, Document Content: {doc.content}")

st.title('DGF Product Seeker Bot')

st.sidebar.title('Options')
show_chroma = st.sidebar.checkbox('Inspect Chroma Database')

if show_chroma:
    inspect_chroma()

query = st.text_input("Qu'est ce que vous cherchez? Ex: Laptop avec 16gb de ram")
if query:
    answer = ask_bot(query)
    st.markdown(answer)
