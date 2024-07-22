import argparse
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pandas as pd
import os

CHROMA_PATH = "chroma"
DOCUMENTS_PATH = "stored_documents.pkl"  # Path to store the list of documents

def store_documents(documents):
    """Store the documents in a file."""
    import pickle
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump(documents, f)

def load_stored_documents():
    """Load the documents from the file."""
    import pickle
    with open(DOCUMENTS_PATH, 'rb') as f:
        return pickle.load(f)

def extract_marque_values():
    """Extract and display the 'marque' values from the stored documents."""
    documents = load_stored_documents()
    
    # Extract and print the first 10 'marque' values
    marque_values = [doc.metadata['marque'] for doc in documents]
    print("First 10 'marque' values:", marque_values[:10])

def search(query_text, k=3):
    """Perform a similarity search and generate a response."""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    
    # Debugging: print raw results
    print("Raw Results:", results)
    
    normalized_results = normalize_scores(results)  # Normalize the scores
    
    # Debugging: print normalized results
    print("Normalized Results:", normalized_results)

    if len(normalized_results) == 0 or normalized_results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in normalized_results])
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=500,
        top_p=1,
        stream=False,
        stop=None,
    )
    response_text = response.choices[0].message['content']

    sources = [doc.metadata.get("source", None) for doc, _score in normalized_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, nargs='?', help="The query text.", default=None)
    args = parser.parse_args()
    query_text = args.query_text

    if query_text:
        # Perform search and generate response.
        search(query_text)
    else:
        # Extract and display 'marque' values
        extract_marque_values()

if __name__ == "__main__":
    main()
