# E-Commerce Recommendation System

AI-powered FastAPI-based recommendation system that delivers personalized top-5 product suggestions with human-like explanations. It uses product embeddings, user interactions, LLMs, and sentiment analysis for natural recommendations.

# Accuracy 
Got 90.4% recommendation similarity accuracy 
Avg accuracy for all recommendations is 80-93%

# 🚀 Features
## ✅ Personalized Recommendations

Uses cosine similarity on all-MiniLM-L6-v2 embeddings based on purchases, views, and interactions.

## ✅ Conversational Explanations (~100 words)

Generated using TinyLLaMA, Mistral, or Flan-T5.( Multiple AIs for fallback)
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
git clone https://github.com/rasikaphutane/ecom.git
cd ecommerce-recommendation

2. Install Dependencies
pip install requirements.txt

4. Set Hugging Face Token

5. Start Ollama (for Mistral)
ollama serve ( not needed but will be easier to monitor)
ollama pull mistral

## ▶️ Run the FastAPI Server

## Products Check 
Enter the type and no. of items to get a list

## Get Recommendations
curl -X POST http://localhost:[portno.]/recommendations \
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

## To run locally

Generate the recommendations locally by running 
python generate_recom.py

