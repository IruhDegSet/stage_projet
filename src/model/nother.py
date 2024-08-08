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
    # Configuration des embeddings et de la base de données
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = qd.from_existing_collection(
        embedding=embeddings,
        url='https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333',
        prefer_grpc=True,
        api_key='lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA',
        collection_name="lvHP_collection",
        vector_name=''
    )

    # Configuration du modèle et du prompt
    llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
    template = """Vous êtes un assistant vendeur. Vous avez accès uniquement au contexte fourni et à 
    l'historique des questions et réponses. Ne générez pas d'informations si elles ne sont pas dans 
    le contexte. Répondez seulement si vous avez la réponse. Accompagnez chaque réponse du numéro de référence,
      de la marque et de la description du produit tel qu'ils sont dans le contexte. Affichez autant de lignes que 
      les produits trouvés dans le contexte. Répondez à la question de l'utilisateur en français. Vous êtes
    obligé de répondre dans un tableau avec les colonnes suivantes : référence, marque et description.
     si je te pose une question sur les questions ou les réponses fournies précédemment, tu dois me répondre selon l'historique.
    tu ne dois pas oublier l'historique car parfois l'utilisateur continue à poser des questions sur tes réponses déjà fournies auparavant.

    Contexte :
    {context}

    Question : {question}
        Historique des questions et réponses :
   {historique}
    Réponse :"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Initialisation de la chaîne QA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k': k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    print("Configuration de la chaîne QA : ", qa_chain)
    # Chargement de l'historique
    conversation_history = memory.load_memory_variables({})
    
    # Vérification si 'history' existe dans l'historique chargé
    historique = conversation_history.get('history', "")
    # Exécution de la chaîne QA
    result = qa_chain.invoke({"query": query,
                "historique":conversation_history['history'],})

    # Mise à jour de la mémoire
    memory.save_context({'input': query}, {"output": result['result']})
    st.write("Mémoire mise à jour :", memory.load_memory_variables({}))

    return result['result']

st.title('DGF Product Seeker Bot')
query = st.chat_input("Qu'est-ce que vous cherchez ? Ex : Laptop avec 16 Go de RAM")
if query:
    answer = ask_bot(query)
    st.markdown(answer)
