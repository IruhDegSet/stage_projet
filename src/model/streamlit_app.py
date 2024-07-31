# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
try:
    from langchain_community.document_loaders.csv_loader import CSVLoader
except ImportError:
    from langchain.document_loaders.csv_loader import CSVLoader  # Alternative import

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
<<<<<<< HEAD
from langchain_qdrant import QdrantVectorStore as qd
=======
from langchain_community.vectorstores import Chroma
>>>>>>> 3c65813bc94dfe3d694643c400c2a8305d89c3f6
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
CHROMA_PATH = "../data/chroma"
BATCH_SIZE = 1000

COLLECTION_CSV = 'csv_collection'
MBD_MODEL = 'intfloat/multilingual-e5-large'

<<<<<<< HEAD

=======
>>>>>>> 3c65813bc94dfe3d694643c400c2a8305d89c3f6
def ask_bot(query: str, k: int = 10):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = qd.from_existing_collection(embedding=embeddings,
    url='https://c5cf721b-9899-48a6-b9f5-98f015b29a14.us-east4-0.gcp.cloud.qdrant.io:6333',
    prefer_grpc=True,
    api_key='N5ynmJyHHD4Da9pD5sTQjBgAVXR4vt0NN57k9Q8AqzsYOrgk4_Q4cg',
    collection_name="icecat_200k",)

    # Initiate model
    llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)

    # Build prompt
    template = """tu es un assistant vendeur, tu as acces au context seulement. ne generes pas des infos si elles ne sont pas dans le context il faut repondre seulement si tu as la reponse. accompagne chaque reponse du part, marque et description du produit tel qu'ils sont dans le context. affiche autant de lignes que les produit trouves dans le context. repond a la question de l'utilisateur en francais. tu est oblige de repondre dans un tableau avec comme colonnes: reference, marque et la description
    {context}
    Question: {question}
    Reponse:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Build chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k': k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

<<<<<<< HEAD
    
    # Run chain:
    result = qa_chain.invoke({"query": query})
    # st.write(result) 
=======
    # Run chain:
    result = qa_chain.invoke({"query": query})
>>>>>>> 3c65813bc94dfe3d694643c400c2a8305d89c3f6
    return result['result']

st.title('DGF Product Seeker Bot')
query = st.chat_input("Qu'est ce que vous cherchez? Ex: Laptop avec 16gb de ram")
if query:
<<<<<<< HEAD
    # inspect_retriever(query)
=======
>>>>>>> 3c65813bc94dfe3d694643c400c2a8305d89c3f6
    answer = ask_bot(query)
    st.markdown(answer)
