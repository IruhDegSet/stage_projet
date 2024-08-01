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

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
CHROMA_PATH = "../data/chroma"
BATCH_SIZE = 1000
COLLECTION_CSV = 'csv_collection'
MBD_MODEL = 'intfloat/multilingual-e5-large'

# Initialize memory and conversation chain globally
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(
    llm=ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0),
    memory=memory,
    verbose=True
)

def ask_bot(query: str, k: int = 10):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = qd.from_existing_collection(
        embedding=embeddings,
        url='https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333',
        prefer_grpc=True,
        api_key='lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA',
        collection_name="icecat_collection",
        vector_name='vector_params',
    )

    # Build prompt
    template = """Tu es un assistant vendeur, tu as accès au contexte seulement. Ne génères pas d'infos si elles ne sont pas dans le contexte. Il faut répondre seulement si tu as la réponse. Accompagne chaque réponse du part, marque et description du produit tel qu'ils sont dans le contexte. Affiche autant de lignes que les produits trouvés dans le contexte. Réponds à la question de l'utilisateur en français. Tu es obligé de répondre dans un tableau avec comme colonnes : référence, marque et description.
    {context}
    Question: {question}
    Réponse:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Build chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=conversation_chain.llm,
        retriever=vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k': k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    
    # Run chain
    result = qa_chain.invoke({"query": query})

    # Ensure result['result'] is a string
    response_text = str(result.get('result', ''))

    # Save context
    memory.save_context({'input': query}, {"output": response_text})

    return response_text

st.title('DGF Product Seeker Bot')
query = st.chat_input("Qu'est-ce que vous cherchez ? Ex : Laptop avec 16 Go de RAM")
if query:
    answer = ask_bot(query)
    st.markdown(answer)
