from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

print("Vérification Jour 1...\n")

# Test 1 : Qdrant accessible
try:
    client = QdrantClient(url="http://localhost:6333")
    client.get_collections()
    print("✅ Qdrant : connecté sur localhost:6333")
except Exception as e:
    print(f"❌ Qdrant : {e}")

# Test 2 : BGE-M3 chargeable
try:
    model = SentenceTransformer("BAAI/bge-m3")
    vec = model.encode("énergie solaire Tunisie")
    print(f"✅ BGE-M3 : chargé — dimension vecteur = {len(vec)}")
except Exception as e:
    print(f"❌ BGE-M3 : {e}")

print("\nJour 1 terminé si tous les ✅ sont verts !")
