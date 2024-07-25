import chromadb

def list_chroma_collections(chroma_path):
    try:
        # Initialiser le client Chroma
        db = chromadb.PersistentClient(path=chroma_path)
        
        # Récupérer toutes les collections
        collections = db.list_collections()
        
        # Afficher les noms des collections
        if collections:
            print("Collections in Chroma DB:")
            for collection in collections:
                print(collection)  # Affiche chaque nom de collection
        else:
            print("No collections found in Chroma DB.")
            
        # Afficher le nombre de collections
        print("Number of collections:", len(collections))
    
    except Exception as e:
        print(f"An error occurred while listing collections: {e}")

# Exemple d'utilisation
list_chroma_collections("../data/chroma")
