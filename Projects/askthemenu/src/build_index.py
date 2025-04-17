import numpy as np
import faiss

embeddings = np.load('data/embeddings.npy')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, 'data/faiss_index.index')
print("✅ Saved FAISS index to data/faiss_index.index")
