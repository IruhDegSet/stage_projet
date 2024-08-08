import streamlit as st
try:
    from langchain_community.document_loaders.csv_loader import CSVLoader
except ImportError:
    from langchain.document_loaders.csv_loader import CSVLoader  # Alternative import

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore as qd
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
CHROMA_PATH = "../data/chroma"
BATCH_SIZE = 1000

COLLECTION_CSV = 'csv_collection'
MBD_MODEL = 'intfloat/multilingual-e5-large'
memory = ConversationBufferMemory()
def ask_bot(query: str, k: int = 10):
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = qd.from_existing_collection(embedding=embeddings,
    url='https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333',
    prefer_grpc=True,
    api_key= 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA',
    collection_name="lvHP_collection",
    vector_name='',)

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


    conversation_history = memory.load_memory_variables({})
    st.write("Conversation History Loaded:", conversation_history)
    # Execute QA chain
    result = qa_chain.invoke({"query": query, "historique": conversation_history.get('history', [])})
    memory.save_context({'input': query}, {"output": result['result']})
    st.write("Memory Updated:", memory.load_memory_variables({}))
    return result['result']

st.title('DGF Product Seeker Bot')
query = st.chat_input("Qu'est ce que vous cherchez? Ex: Laptop avec 16gb de ram")
if query:

    # inspect_retriever(query)
    answer = ask_bot(query)
    st.markdown(answer)
