import gradio as gr
from langchain_community.document_loaders.csv_loader import CSVLoader
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

    # Run chain:
    result = qa_chain.invoke({"query": query})
    return result['result']

# Create a Gradio interface
interface = gr.Interface(
    fn=ask_bot,
    inputs=gr.Textbox(lines=2, placeholder="Qu'est ce que vous cherchez? Ex: Laptop avec 16gb de ram"),
    outputs="markdown",
    title="DGF Product Seeker Bot",
    description="Ask the bot about products and get a detailed response."
)

# Launch the Gradio app
interface.launch(share=True)