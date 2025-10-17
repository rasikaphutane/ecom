from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import requests
import time
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="E-Commerce Recommendation API",
    description="AI-powered product recommendations with behavior-based explanations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "mistral:7b-instruct-q4_K_M"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global variables
embedder = None
products = None
users = None
interactions = None
product_embeddings = None
product_id_to_idx = None
valid_product_ids = None

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 5
    min_rating: float = 4.0

class ProductRecommendation(BaseModel):
    productId: int
    productName: str
    productCategory: str
    productDescription: str
    avgRating: float
    explanation: str

class UserBehaviorResponse(BaseModel):
    user_id: int
    total_purchases: int
    total_views: int
    top_categories: List[str]
    purchased_products: List[str]
    avg_purchase_rating: float

class HealthCheck(BaseModel):
    status: str
    ollama_available: bool
    database_connected: bool
    models_loaded: bool
    timestamp: str

# Initialize the system
@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup."""
    global embedder, products, users, interactions, product_embeddings, product_id_to_idx, valid_product_ids
    
    logger.info("🚀 Initializing Recommendation System...")
    
    try:
        # Load embedding model
        embedder = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
        logger.info("✅ SentenceTransformer loaded")
        
        # Load database
        products, users, interactions = load_database()
        logger.info(f"✅ Database loaded: {len(products)} products, {len(users)} users")
        
        # Generate embeddings
        product_embeddings = generate_embeddings(products)
        logger.info("✅ Product embeddings generated")
        
        # Create mappings
        product_id_to_idx = {pid: idx for idx, pid in enumerate(products['productId'].values)}
        valid_product_ids = set(products['productId'].values)
        logger.info("✅ System mappings created")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

def load_database():
    """Load and validate database."""
    try:
        conn = sqlite3.connect('ecommerce.db')
        
        products = pd.read_sql('SELECT * FROM products', conn)
        users = pd.read_sql('SELECT * FROM users', conn)
        interactions = pd.read_sql('SELECT * FROM interactions', conn)
        
        conn.close()
        
        if products.empty or users.empty:
            raise ValueError("Database tables are empty")
            
        return products, users, interactions
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise

def generate_embeddings(products_df):
    """Generate product embeddings."""
    product_descriptions = products_df.apply(
        lambda row: f"{row['productName']} {row['productCategory']} {row['productDescription']}",
        axis=1
    ).tolist()
    
    embeddings = embedder.encode(
        product_descriptions,
        convert_to_tensor=True,
        device=DEVICE,
        show_progress_bar=False
    )
    
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().numpy()
        
    return embeddings

# Helper functions
def test_ollama_connection():
    """Test if Ollama is available."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_explanation(prompt: str) -> str:
    """Generate explanation using Ollama."""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "top_p": 0.85,
                "num_predict": 120,
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            explanation = result.get('response', '').strip()
            explanation = re.sub(r'\s+', ' ', explanation)
            return explanation
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        return "This product is recommended based on your shopping behavior and preferences."

def get_user_behavior(user_id: int) -> Dict[str, Any]:
    """Get user behavior analysis."""
    try:
        conn = sqlite3.connect('ecommerce.db')
        
        purchase_query = """
        SELECT p.productId, p.productName, p.productCategory, p.avgRating
        FROM interactions i
        JOIN products p ON i.productId = p.productId
        WHERE i.userId = ? AND i.boughtStatus = 1
        ORDER BY p.productId DESC
        LIMIT 10
        """
        user_purchases = pd.read_sql(purchase_query, conn, params=(user_id,))
        
        view_query = """
        SELECT p.productId, p.productName, p.productCategory, COUNT(*) as view_count
        FROM interactions i
        JOIN products p ON i.productId = p.productId
        WHERE i.userId = ? AND i.viewStatus = 1
        GROUP BY p.productId
        ORDER BY view_count DESC
        LIMIT 10
        """
        user_views = pd.read_sql(view_query, conn, params=(user_id,))
        
        conn.close()
        
        purchase_categories = user_purchases['productCategory'].value_counts().to_dict()
        avg_rating = user_purchases['avgRating'].mean() if not user_purchases.empty else 0
        
        return {
            'purchases': user_purchases.to_dict('records'),
            'views': user_views.to_dict('records'),
            'purchase_categories': purchase_categories,
            'total_purchases': len(user_purchases),
            'total_views': len(user_views),
            'avg_purchase_rating': avg_rating,
            'top_categories': list(purchase_categories.keys())[:3],
            'purchased_product_names': user_purchases['productName'].tolist()[:3]
        }
        
    except Exception as e:
        logger.error(f"Error getting user behavior: {e}")
        return {
            'purchases': [], 'views': [], 'purchase_categories': {},
            'total_purchases': 0, 'total_views': 0, 'avg_purchase_rating': 0,
            'top_categories': [], 'purchased_product_names': []
        }

# API Routes
@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint with health check."""
    ollama_ok = test_ollama_connection()
    db_ok = products is not None and not products.empty
    models_ok = embedder is not None and product_embeddings is not None
    
    status = "healthy" if all([ollama_ok, db_ok, models_ok]) else "degraded"
    
    return HealthCheck(
        status=status,
        ollama_available=ollama_ok,
        database_connected=db_ok,
        models_loaded=models_ok,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return await root()

@app.get("/users/{user_id}/behavior", response_model=UserBehaviorResponse)
async def get_user_behavior_endpoint(user_id: int):
    """Get user behavior analysis."""
    behavior = get_user_behavior(user_id)
    
    return UserBehaviorResponse(
        user_id=user_id,
        total_purchases=behavior['total_purchases'],
        total_views=behavior['total_views'],
        top_categories=behavior['top_categories'],
        purchased_products=behavior['purchased_product_names'],
        avg_purchase_rating=behavior['avg_purchase_rating']
    )

@app.post("/recommendations", response_model=List[ProductRecommendation])
async def get_recommendations(request: RecommendationRequest):
    """Get personalized product recommendations."""
    try:
        logger.info(f"🎯 Generating recommendations for user {request.user_id}")
        
        # Validate user exists
        user = users[users['userId'] == request.user_id]
        if user.empty:
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        
        # Parse user interactions
        interactions_str = user['interactions'].iloc[0] if 'interactions' in user.columns else ''
        if not interactions_str:
            raise HTTPException(status_code=404, detail=f"No interactions found for user {request.user_id}")
        
        try:
            user_interactions = [int(x.strip()) for x in str(interactions_str).split(',') if x.strip().isdigit()]
            user_interactions = [pid for pid in user_interactions if pid in valid_product_ids]
        except:
            user_interactions = []
        
        if not user_interactions:
            raise HTTPException(status_code=404, detail=f"No valid interactions for user {request.user_id}")
        
        # Get user behavior
        user_behavior = get_user_behavior(request.user_id)
        
        # Calculate user embedding
        user_product_indices = []
        for pid in user_interactions:
            if pid in product_id_to_idx:
                user_product_indices.append(product_id_to_idx[pid])
        
        if not user_product_indices:
            raise HTTPException(status_code=404, detail="No valid product indices found")
        
        user_emb = np.mean(product_embeddings[user_product_indices], axis=0)
        sim_scores = cosine_similarity([user_emb], product_embeddings)[0]
        
        # Generate recommendations
        recommendations = []
        candidate_indices = np.argsort(sim_scores)[::-1]
        
        for idx in candidate_indices:
            if len(recommendations) >= request.top_k:
                break
                
            product_id = products.iloc[idx]['productId']
            if product_id in user_interactions:
                continue
                
            product_rating = products.iloc[idx]['avgRating']
            if product_rating < request.min_rating:
                continue
            
            # Generate explanation
            prompt = f"""Explain why {products.iloc[idx]['productName']} is recommended to this user.

User purchased: {user_behavior['purchased_product_names']}
User likes categories: {user_behavior['top_categories']}
Product: {products.iloc[idx]['productName']} ({products.iloc[idx]['productCategory']})
Rating: {product_rating}/5

Write a concise, personalized explanation."""
            
            explanation = generate_explanation(prompt)
            
            recommendation = ProductRecommendation(
                productId=product_id,
                productName=products.iloc[idx]['productName'],
                productCategory=products.iloc[idx]['productCategory'],
                productDescription=products.iloc[idx]['productDescription'],
                avgRating=product_rating,
                explanation=explanation
            )
            
            recommendations.append(recommendation)
        
        logger.info(f"✅ Generated {len(recommendations)} recommendations for user {request.user_id}")
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/products", response_model=List[Dict[str, Any]])
async def get_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_rating: Optional[float] = Query(None, description="Minimum rating"),
    limit: int = Query(50, description="Number of products to return")
):
    """Get products with filtering."""
    try:
        filtered_products = products.copy()
        
        if category:
            filtered_products = filtered_products[filtered_products['productCategory'] == category]
        
        if min_rating is not None:
            filtered_products = filtered_products[filtered_products['avgRating'] >= min_rating]
        
        return filtered_products.head(limit).to_dict('records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)