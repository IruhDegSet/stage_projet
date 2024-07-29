import time
import pandas as pd
import re
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from groq import Groq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "../data/chroma"
GROQ_TOKEN = os.getenv('GROQ_TOKEN')
HF_TOKEN = os.getenv('API_TOKEN')

PROMPT_TEMPLATE = """
la qst:

{question}

---

la reponse: {context}
"""

def get_embedding_function(model_name: str = "intfloat/multilingual-e5-large"):
    """Get Hugging Face embeddings function."""
    return HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name=model_name)

def get_groq_response(prompt_text: str) -> str:
    """Generate a response using Groq API."""
    client = Groq(api_key=GROQ_TOKEN)

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    # Accéder au contenu de la réponse de manière appropriée
    response_message = completion.choices[0].message
    response_content = response_message.content if hasattr(response_message, 'content') else str(response_message)
    
    return response_content

def query_rag(query_text: str, file_type: str) -> str:
    try:
        start_time = time.time()
        
        # Préparer la fonction d'embedding
        embedding_function = get_embedding_function()

        # Charger la base de données vectorielle avec la collection appropriée
        collection_name = "txt_collection" if file_type == "txt" else "csv_collection"
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_name=collection_name)

        # Utiliser Chroma pour la recherche de similarité
        results = db.similarity_search(query_text, k=50)  # Par exemple, pour obtenir jusqu'à 20 résultats

        if not results:
            print("No documents found.")
            return "No documents found."

        # Les résultats contiennent des documents et des scores de similarité
        documents_with_scores = [(doc, doc.metadata.get("score", 0)) for doc in results]

        # Trier les documents par score de similarité (du plus élevé au plus bas)
        documents_with_scores.sort(key=lambda x: x[1], reverse=True)
  
        # Préparer le texte du contexte à partir des documents top-k
        top_k_results = documents_with_scores[:10]
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in top_k_results])

        # Préparer le prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Générer la réponse en utilisant Groq
        response_text = get_groq_response(prompt)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for query: {elapsed_time:.2f} seconds")
        
        # Nettoyer et formater la réponse
        # Assurez-vous que la réponse ne contient pas de doublons et est bien formatée
        return response_text.strip()  # Utiliser strip() pour enlever les espaces inutiles

    except Exception as e:
        import traceback
        print(f"An error occurred in query_rag: {e}")
        print(traceback.format_exc())
