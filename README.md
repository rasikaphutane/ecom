# E-Commerce Recommendation System

AI-powered FastAPI-based recommendation system that delivers personalized top-5 product suggestions with human-like explanations. It uses product embeddings, user interactions, LLMs, and sentiment analysis for natural recommendations.

# Accuracy 
Got 90.4% recommendation similarity accuracy 
Avg accuracy for all recommendations is 80-93%

# 🚀 Features
## ✅ Personalized Recommendations

Uses cosine similarity on all-MiniLM-L6-v2 embeddings based on purchases, views, and interactions.

## ✅ Conversational Explanations (~100 words)

Generated using TinyLLaMA, Mistral, or Flan-T5.
Example: “Hey, this Sweater is perfect for you!”

## ✅ Sentiment Analysis

Uses DistilBERT with keyword fallback (e.g., “great,” “cozy”).

## ✅ FastAPI Endpoints

GET / – Health Check

GET /users/{user_id}/behavior

POST /recommendations

GET /products

## ✅ Database

SQLite ecommerce.db with ~50 products, 30 users, and 100 interactions.

# 🧠 Tech Stack

Backend: FastAPI, Uvicorn, Pydantic

Embeddings: all-MiniLM-L6-v2

LLMs: TinyLLaMA, Mistral, Flan-T5

Sentiment: DistilBERT + keyword fallback

Database: SQLite

Server: Ollama

# 🔧 Installation
1. Clone the Repo
git clone <repository-url>
cd ecommerce-recommendation

2. Install Dependencies
pip install fastapi uvicorn pandas numpy sentence-transformers torch transformers scikit-learn requests nltk
python -m nltk.downloader punkt punkt_tab

3. Set Hugging Face Token
Windows: 
set HUGGINGFACE_TOKEN=hf_xxx

Linux/Mac: 
export HUGGINGFACE_TOKEN='hf_xxx'

4. Initialize the Database
python init_db.py
python generate_data.py

5. Start Ollama (for Mistral)
ollama run mistral:7b-instruct-q4_K_M

## ▶️ Run the FastAPI Server
uvicorn main:app --host 0.0.0.0 --port 1234

## 🌐 API Access
Swagger UI
http://localhost:1234/docs

Health Check
curl http://localhost:1234/

Get Recommendations
curl -X POST http://localhost:1234/recommendations \
-H "Content-Type: application/json" \
-d '{"user_id": 1, "top_k": 5, "min_rating": 4.0}'

✅ Example Output
{
  "productId": 19,
  "productName": "Sweater",
  "productCategory": "Clothing",
  "productDescription": "Comfortable Sweater with premium fabric",
  "avgRating": 4.5,
  "explanation": "Hey, I think this Sweater is perfect for you! You’ve been eyeing comfy clothes like sweaters..."
}

# feel free to use/modify!!
📜 License

MIT License – free to use and modify.


