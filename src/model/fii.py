import pandas as pd
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
import requests

# Constants
API_TOKEN = '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'
MBD_MODEL = 'intfloat/multilingual-e5-large'
DRANT_URL = "https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333"
COLLECTION_NAME = "collection_icecat"
QDRANT_API_KEY = 'lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA'
# Charger les données du CSV
df = pd.read_csv('../data/icecat.csv')

# Initialiser les embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=API_TOKEN, model_name=MBD_MODEL)

# Vérifier le nombre de documents dans la collection Qdrant
headers = {
    'api-key': QDRANT_API_KEY,
    'Content-Type': 'application/json'
}

url = f'{DRANT_URL}/collections/{COLLECTION_NAME}'

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    num_points = data['result']['points_count']
    print(f'Number of documents in the collection: {num_points}')
else:
    print(f'Error: {response.status_code}')
    print(response.text)
