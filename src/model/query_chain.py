from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceHubEmbeddings

# CONSTS
API_TOKEN='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'
GROQ_TOKEN='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'
CHROMA_PATH = "../data/chroma"
BATCH_SIZE = 1000
RAW_DATA_PATH= '../data/data_base.csv'
FEATURES= ['part', 'marque', 'description']
DATA_PATH_CSV= 'data/first_1000_lines.csv'
DATA_PATH_TXT= 'data/sample_db.txt'
COLLECTION_TXT= 'txt_collection'
COLLECTION_CSV= 'csv_collection'
MBD_MODEL= 'intfloat/multilingual-e5-large'


persist_directory = CHROMA_PATH
# vectordb.delete_collection()
embedding = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL, )
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=COLLECTION_CSV)

#initiate model
llm = ChatGroq(model_name='llama3-8b-8192', api_key= GROQ_TOKEN,temperature=0)

# Build prompt
template = """tu es un assistant vendeur, tu as acces au context seulement. ne generes pas des infos si ell ne sont pas dans le context il faut repondre seulement si tu as la reponse. accompagne chaque reponse du part, marque et description. affiche autant de lignes que les produit trouve dans le context. repond a la question de l'utilisateur en francais 
{context}
Question: {question}
Reponse:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever( search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    verbose= True
)

# test chain: 
question = "I want a microphone with noise cancelling"
result = qa_chain({"query": question})
print(result["result"])