import streamlit as st
import uuid
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore as qd
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
CHROMA_PATH = "../data/chroma"
COLLECTION_CSV = 'csv_collection'
MBD_MODEL = 'intfloat/multilingual-e5-large'

def initialize_store():
    if 'store' not in st.session_state:
        st.session_state.store = {}

def get_session_id():
    # Générer un identifiant de session basé sur les sessions Streamlit
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def ask_bot(query: str, k: int = 10):
    initialize_store()
    session_id = get_session_id()  # Utiliser l'identifiant de session Streamlit
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = qd.from_existing_collection(
        embedding=embeddings,
        url='https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333',
        prefer_grpc=True,
        api_key='lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA',
        collection_name="lvHP_collection",
        vector_name=''
    )
    
    llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
   
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k': k})

    contextualize_q_system_prompt = (
        "Étant donné un historique de conversation et la dernière question de l'utilisateur "
        "qui pourrait faire référence au contexte de l'historique de conversation, "
        "formulez une question autonome qui peut être comprise "
        "sans l'historique de conversation. NE RÉPONDEZ PAS à la question, "
        "reformulez-la si nécessaire et, sinon, renvoyez-la telle quelle."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    template = """
        Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
        Répond seulement si tu as la réponse. Affiche les produits un par un sous forme de tableau qui contient ces colonnes Référence, Categorie, Marque, Description.
        Il faut savoir que laptop, ordinateur, ordinateurs portables, pc et poste de travail ont tous le même sens.
        Il faut savoir que téléphone portable et smartphone ont le même sens.
        Il faut savoir que tout autre caractéristique du produit tel que la RAM et le stockage font partie de la description du produit et il faut filtrer selon la marque et la catégorie seulement.
        Si le contexte est vide, dis-moi que tu n'as pas trouvé de produits correspondants. Je veux que la réponse soit claire et facile à lire, avec des sauts de ligne pour séparer chaque produit. Ne me donne pas de produits qui ne sont pas dans le contexte.
        Si je te pose une question sur les questions ou les réponses fournies précédemment, tu dois me répondre selon l'historique.
        Tu ne dois pas oublier l'historique car parfois l'utilisateur continue à te poser des questions sur tes réponses que tu as déjà fournies auparavant.
        Contexte: {context}
        Réponse :
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    st.write(f"Session ID: {session_id}")
    st.write("Store Before:", st.session_state.store)

    response = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id}
        },
    )

    st.write("Store After:", st.session_state.store)
    st.write("Response:", response["answer"])

    return response["answer"]

st.title('DGF Product Seeker Bot')
query = st.chat_input("Qu'est-ce que vous cherchez ? Ex : Laptop avec 16 Go de RAM")
if query:
    answer = ask_bot(query)  # Receiving only one value
    st.markdown(answer)
