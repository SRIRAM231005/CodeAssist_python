import os
import numpy as np
import chromadb
from tqdm import tqdm

os.makedirs("./chroma_store", exist_ok=True)

embeddings = np.load("data/codebert_embeddings.npy")   # shape: (N, 768)
metadata = np.load("data/metadata.npy", allow_pickle=True)  # shape: (N,)

assert len(embeddings) == len(metadata), "Embeddings and metadata size mismatch"

chroma_client = chromadb.PersistentClient(
    path="./chroma_store"
)

collection = chroma_client.get_or_create_collection(
    name="codebert_python"
)

BATCH_SIZE = 500

for i in tqdm(range(0, len(embeddings), BATCH_SIZE)):
    batch_embeddings = embeddings[i:i + BATCH_SIZE]
    batch_meta = metadata[i:i + BATCH_SIZE]

    ids = [f"code_{i + j}" for j in range(len(batch_embeddings))]

    documents = [
        m.get("summary") or m.get("docstring") or "Code function description."
        for m in batch_meta
    ]

    metadatas = [
        {
            "repo": m.get("repo", ""),
            "path": m.get("path", ""),
            "func_name": m.get("func_name", ""),
            "docstring": m.get("docstring", "")
        }
        for m in batch_meta
    ]

    collection.add(
        ids=ids,
        embeddings=batch_embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )

#chroma_client.persist()

print("✅ All embeddings and metadata stored in ChromaDB")
print("Total vectors in DB:", collection.count())
