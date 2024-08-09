import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore as qd
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Constants
API_TOKEN = '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'
GROQ_TOKEN = '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'
CHROMA_PATH = "../data/chroma"
COLLECTION_CSV = 'csv_collection'
MBD_MODEL = 'intfloat/multilingual-e5-large'


def ask_bot(question: str, k: int = 10):
    st.write('query:', question)
    # Initialisation de la mémoire
    memory = ConversationBufferMemory(
        input_key='query',
        memory_key="history",
    )

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
    template = """
                    Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
                    Répond seulement si tu as la réponse. Affiche les produits un par un sous forme de tableau qui contient ces colonne Référence,Categorie, Marque, Description.
                    Il faut savoir que laptop, ordinateur, ordinateurs portable , pc et poste de travail ont tous le même sens.
                    Il faut savoir que téléphone portable et smartphone ont le même sens.
                    Il faut savoir que tout autre caractéristique du produit tel que la RAM stockage font partie de la description du produit et il faut filtrer selon la marque et la catégorie seulement.
                
                    Si le contexte est vide, dis-moi que tu n'as pas trouvé de produits correspondants. Je veux que la réponse soit claire et facile à lire, avec des sauts de ligne pour séparer chaque produit. Ne me donne pas de produits qui ne sont pas dans le contexte.
                    si je te pose une question sur les question ou les reponses fournients précédemment tu dois me répondre selon l'historique.
                    tu ne dois pas oublier l'historique car parfois le user continue à te poser des questions sur tes réponses que tu as déjà fournies auparavant
    
                    Contexte: {context}
                    Historique : {history}
                    Question: {query}

                    Réponse :
                    """
  
    prompt = PromptTemplate(
        input_variables=["context", "history", "query"],
        template=template,
    )

    chain_type_kwargs = {
        "prompt": prompt,
        "memory": memory,
    }
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 50, 'fetch_k': k}),
        verbose=True,
        memory=memory,
        chain_type_kwargs=chain_type_kwargs
    )
    st.write(qa)
    # Chargement de l'historique
    conversation_history = memory.load_memory_variables({})
    history = conversation_history.get('history', "")

    # Préparation des inputs avec les clés correctes
    inputs = {
        "context": "",  # Contexte vide par défaut
        "history": history,
        "query": question  # Utilisation de 'query' comme spécifié dans le PromptTemplate
    }

    st.write("Inputs fournis à la chaîne:", inputs)
    
    # Appel à la chaîne avec les clés appropriées
    try:
        result = qa.invoke(inputs)
    except ValueError as e:
        st.error(f"Erreur de valeur : {e}")
        return 'Erreur lors de l\'appel de la chaîne.', []

    # Gestion des résultats
    output = result.get('result', 'Pas de réponse trouvée')
    sources = result.get('source_documents', [])

    # Mise à jour de la mémoire
    memory.save_context({'query': question}, {'result': output})

    return output, sources

st.title('DGF Product Seeker Bot')
query = st.chat_input("Qu'est-ce que vous cherchez ? Ex : Laptop avec 16 Go de RAM")
if query:
    answer, sources = ask_bot(query)
    st.markdown(answer)
    if sources:
        st.write("Sources :")
        for doc in sources:
            st.write(doc)
