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
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def ask_bot(query: str, k: int = 10):
    initialize_store()
    session_id = get_session_id()
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)
    vectordb = qd.from_existing_collection(
        embedding=embeddings,
        url='https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333',
        prefer_grpc=True,
        api_key='lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA',
        collection_name="collection_icecat",
        vector_name=''
    )
    
    llm = ChatGroq(model_name='llama-3.1-70b-versatile', api_key=GROQ_TOKEN, temperature=0)
   
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k': k})

    contextualize_q_system_prompt = (
        "Étant donné un historique de conversation et la dernière question de l'utilisateur "
        "qui pourrait faire référence au contexte de l'historique de conversation, "
        "formulez une question autonome qui peut être comprise "
        "sans l'historique de conversation. NE RÉPONDEZ PAS à la question, "
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
        Tu es un assistant vendeur spécialisé dans les produits de notre base de données. Voici les règles pour répondre aux questions :

        1. **Contexte et Réponse** : Tu as accès uniquement au contexte fourni et aux informations disponibles dans notre base de données. Ne génère pas de nouvelles informations si elles ne sont pas présentes dans le contexte.

        2. **Affichage des Produits** : Lorsque tu fournis des réponses, affiche les produits sous forme de tableau avec les colonnes suivantes :
        - **Référence** : L'identifiant unique du produit.
        - **Catégorie** : La catégorie du produit (ex. ordinateur, téléphone, etc.).
        - **Marque** : La marque du produit.
        - **Description** : Une description détaillée du produit, y compris des caractéristiques comme la RAM et le stockage, si elles sont disponibles.

        3. **Synonymes** : Les termes suivants doivent être considérés comme équivalents :
        - *Laptops, PC/postes de travail, PC tout en un/stations de travail, ordinateur, ordinateurs portables, PC, poste de travail*
        - *Téléphone portable et smartphone*

        4. **Filtrage** : Filtre les produits selon la marque et la catégorie seulement. Les autres caractéristiques comme la RAM et le stockage font partie de la description et ne doivent pas être utilisées pour le filtrage.

        5. **Contexte Vide** : Si le contexte ne contient aucun produit correspondant à la requête, informe l'utilisateur qu'aucun produit correspondant n'a été trouvé.

        6. **Clarté de la Réponse** : La réponse doit être claire et facile à lire. Sépare chaque produit par des sauts de ligne pour une meilleure lisibilité. Ne donne pas de produits qui ne sont pas dans le contexte.

        7. **Historique des Questions** : Si la question se réfère à des questions ou des réponses précédentes, réponds en tenant compte de l'historique des échanges. Assure-toi de ne pas oublier les informations passées, car l'utilisateur peut poser des questions supplémentaires sur des réponses déjà fournies.

        Contexte : {context}
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


    response = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id}
        },
    )

    return response["answer"]

st.title('DGF Product Seeker Bot')
if 'messages' not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Qu'est-ce que vous cherchez ? Ex : Laptop avec 16 Go de RAM")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    answer = ask_bot(query)
    st.session_state.messages.append({"role": "assistant", "content": answer})

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])
