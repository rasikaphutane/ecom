import sqlite3
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load data from SQLite
conn = sqlite3.connect('ecommerce.db')
products = pd.read_sql('SELECT * FROM products', conn)
users = pd.read_sql('SELECT * FROM users', conn)
conn.close()

# Generate embeddings for product descriptions
logger.info("Generating product embeddings...")
product_embeddings = embedder.encode(
    products['productDescription'].tolist(),
    convert_to_tensor=True,
    device=device,
    show_progress_bar=True
).cpu().numpy()  # Shape: (n_products, embedding_dim)

# Create FAISS index
embedding_dim = product_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (equivalent to cosine after normalization)
faiss_index.add(product_embeddings.astype(np.float32))  # Add embeddings to index

def get_recommendations_cosine(user_id, top_k=5):
    """Recommend products using cosine similarity."""
    user = users[users['userId'] == user_id]
    if user.empty:
        logger.warning(f"User {user_id} not found.")
        return []
    
    interactions_str = user['interactions'].values[0]
    if not interactions_str:
        logger.warning(f"No bought interactions for user {user_id}")
        return []
    
    try:
        interactions = [int(x) for x in interactions_str.split(',')]
    except ValueError as e:
        logger.error(f"Error parsing interactions for user {user_id}: {interactions_str}, Error: {e}")
        return []
    
    valid_interactions = [pid for pid in interactions if pid in products['productId'].values]
    if not valid_interactions:
        logger.warning(f"No valid bought interactions for user {user_id}")
        return []
    
    # Compute user embedding (average of bought product embeddings)
    user_emb = np.mean(product_embeddings[products[products['productId'].isin(valid_interactions)].index], axis=0)
    
    # Cosine similarity
    sim_scores = cosine_similarity([user_emb], product_embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1]
    
    # Filter out already bought products
    recs = []
    for idx in top_indices:
        prod_id = products.iloc[idx]['productId']
        if prod_id not in interactions:
            recs.append({'productId': prod_id, 'productName': products.iloc[idx]['productName']})
            if len(recs) == top_k:
                break
    
    return recs

def get_recommendations_faiss(user_id, top_k=5):
    """Recommend products using FAISS L2 distance."""
    user = users[users['userId'] == user_id]
    if user.empty:
        logger.warning(f"User {user_id} not found.")
        return []
    
    interactions_str = user['interactions'].values[0]
    if not interactions_str:
        logger.warning(f"No bought interactions for user {user_id}")
        return []
    
    try:
        interactions = [int(x) for x in interactions_str.split(',')]
    except ValueError as e:
        logger.error(f"Error parsing interactions for user {user_id}: {interactions_str}, Error: {e}")
        return []
    
    valid_interactions = [pid for pid in interactions if pid in products['productId'].values]
    if not valid_interactions:
        logger.warning(f"No valid bought interactions for user {user_id}")
        return []
    
    # Compute user embedding
    user_emb = np.mean(product_embeddings[products[products['productId'].isin(valid_interactions)].index], axis=0)
    
    # FAISS search (L2 distance)
    user_emb = user_emb.astype(np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(user_emb, top_k + len(interactions))  # Query more to filter
    
    # Filter out already bought products
    recs = []
    for idx in indices[0]:
        prod_id = products.iloc[idx]['productId']
        if prod_id not in interactions:
            recs.append({'productId': prod_id, 'productName': products.iloc[idx]['productName']})
            if len(recs) == top_k:
                break
    
    return recs

def get_recommendations(user_id, top_k=5, method='cosine'):
    """Wrapper to select recommendation method."""
    if method == 'cosine':
        return get_recommendations_cosine(user_id, top_k)
    elif method == 'faiss':
        return get_recommendations_faiss(user_id, top_k)
    else:
        raise ValueError("Method must be 'cosine' or 'faiss'")

# Test recommendations
if __name__ == "__main__":
    user_id = 1
    logger.info(f"Testing recommendations for user {user_id}...")
    
    # Test cosine similarity
    recs_cosine = get_recommendations(user_id, top_k=5, method='cosine')
    print(f"\nCosine Recommendations for User {user_id}:")
    print(pd.DataFrame([
        {'userId': user_id, 'productId': rec['productId'], 'productName': rec['productName']}
        for rec in recs_cosine
    ]).to_string(index=False))
    
    # Test FAISS
    recs_faiss = get_recommendations(user_id, top_k=5, method='faiss')
    print(f"\nFAISS Recommendations for User {user_id}:")
    print(pd.DataFrame([
        {'userId': user_id, 'productId': rec['productId'], 'productName': rec['productName']}
        for rec in recs_faiss
    ]).to_string(index=False))