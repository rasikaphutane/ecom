E-Commerce Recommendation System

An AI-powered recommendation system built with FastAPI that delivers personalized top-5 product suggestions along with human-like explanations. It uses product embeddings, user behavior, sentiment analysis, and LLMs to generate engaging, natural recommendations.

🚀 Features

✅ Personalized Recommendations
Uses cosine similarity on product embeddings from all-MiniLM-L6-v2 based on views, purchases, and interactions.

✅ Conversational Explanations (~100 words)
Powered by TinyLLaMA, Mistral, or Flan-T5, e.g.
“Hey, this Sweater is perfect for you!”

✅ Sentiment Analysis
Uses distilbert with a keyword-based fallback (e.g., “great,” “cozy”).

✅ FastAPI Endpoints

GET / – Health check

GET /users/{user_id}/behavior

POST /recommendations

GET /products

✅ Database
Uses ecommerce.db (SQLite) with ~50 products, 30 users, and 100 interactions.

📦 Tech Stack

Backend: FastAPI, Pydantic, Uvicorn

Embeddings: all-MiniLM-L6-v2

LLMs: TinyLLaMA, Mistral, Flan-T5

Sentiment: DistilBERT + keyword fallback

DB: SQLite (ecommerce.db)

Server: Ollama (for Mistral)

🔧 Installation
✅ 1. Clone the Repo
git clone <repository-url>
cd ecommerce-recommendation

✅ 2. Install Dependencies
pip install fastapi uvicorn pandas numpy sentence-transformers torch transformers scikit-learn requests nltk
python -m nltk.downloader punkt punkt_tab

✅ 3. Set Hugging Face Token
# Windows
set HUGGINGFACE_TOKEN=hf_xxx

# Linux/Mac
export HUGGINGFACE_TOKEN='hf_xxx'

✅ 4. Initialize Database
python init_db.py
python generate_data.py

✅ 5. Start Ollama (if using Mistral)
ollama run mistral:7b-instruct-q4_K_M

▶️ Run the FastAPI Server
uvicorn main:app --host 0.0.0.0 --port 1234

✅ API Access

Swagger UI → http://localhost:1234/docs

Health Check:

curl http://localhost:1234/


Get Recommendations:

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
  "explanation": "Hey, I think this Sweater is perfect for you! You’ve been eyeing comfy clothes like sweaters and denim trousers..."
}

🔄 Future Improvements

Diversify product recommendations (reduce Clothing bias)

Fix DistilBERT token errors

Test smaller LLMs (Phi-2, GPT-Neo)

Flask alternative for deployment

🛠 Troubleshooting
Issue	Fix
Short explanations	Increase num_predict / length_penalty
401 Hugging Face errors	Set valid HUGGINGFACE_TOKEN
Repetitive phrases	Tune no_repeat_ngram_size or regex cleanup
DB errors	Verify ecommerce.db exists and loaded
📜 License

MIT License — free to use and modify.
