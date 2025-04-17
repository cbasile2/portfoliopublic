import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('data/restaurants.csv')
embeddings = np.load('data/embeddings.npy')

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embeddings)

df['cluster'] = labels
df.to_csv('data/restaurants_clustered.csv', index=False)
print("✅ Saved clustered data to data/restaurants_clustered.csv")
