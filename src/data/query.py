from langchain_groq import ChatGroq
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import TextEmbedding
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data_base.csv"


def get_db_retriever(mbd_model: str= 'sentence-transformers/all-MiniLM-L6-v2'):

    embedding = HuggingFaceEmbeddings(model_name=mbd_model)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    
    return vectordb.as_retriever()



prompt= hub.pull('rlm/rag-prompt')

llm= ChatGroq(
    temperature=1,
    model="llama3-8b-8192",
    api_key="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"
)

retriever= get_db_retriever()



def format_docs(docs):
    return '\n---------------\n'.join(doc.page_content for doc in docs)

rag_chain= (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#question
rag_chain.invoke("I want a computer with 16gb of ram")