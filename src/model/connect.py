from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionInfo

# Initialiser le client Qdrant
qdrant_client = QdrantClient(
    url="https://a08399e1-9b23-417d-bc6a-88caa066bca4.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="lJo8SY8JQy7W0KftZqO3nw11gYCWIaJ0mmjcjQ9nFhzFiVamf3k6XA",
)

# Définir les paramètres de la collection
vector_params = VectorParams(
    size=1024,  # Remplacez par la taille des vecteurs que vous utilisez
    distance=Distance.COSINE
)

# Nom de la collection
collection_name = "collection_icecat"



# Vérifier les collections existantes
collections = qdrant_client.get_collections()
print("Collections existantes :", collections)

# Vérifier la configuration de la collection nouvellement créée
collection_info = qdrant_client.get_collection(collection_name)
print(f"Configuration de la collection '{collection_name}':", collection_info)
