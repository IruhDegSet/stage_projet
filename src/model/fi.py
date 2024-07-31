import chromadb
# Initialiser le client Chroma en mode lecture seule, si disponible
db = chromadb.PersistentClient(path="../data/chroma_icecat")

# Lister toutes les collections
collections = db.list_collections()

# Afficher la liste des collections
print("Collections in Chroma DB:", collections)

# Optionnel: afficher le nombre de collections
print("Number of collections:", len(collections))
if collections:
    print("Collections in Chroma DB:")
    for collection_name in collections:
        print(f"Processing collection '{collection_name}'...")
        
else:
    print("No collections found in Chroma DB.")

csv = db.get_or_create_collection("csv_collection")

txt = db.get_or_create_collection("txt_collection")


print("There are csv", csv.count(), "in the collection")
print("There are txt", txt.count(), "in the collection")