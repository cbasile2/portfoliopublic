import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv('data/restaurants.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)
np.save('data/embeddings.npy', embeddings)
print("✅ Saved embeddings to data/embeddings.npy")
