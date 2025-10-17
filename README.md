E-Commerce Recommendation System

This AI-powered e-commerce recommendation system, built with FastAPI, delivers personalized top-5 product recommendations with human-like explanations. It uses user behavior (views, purchases, reviews) from ecommerce.db, product embeddings (all-MiniLM-L6-v2), and LLMs (Flan-T5-Base, TinyLLaMA, Mistral) to generate engaging explanations like “Hey, this Sweater is perfect for you!” Sentiment analysis uses distilbert with a keyword fallback. The FastAPI app provides endpoints for recommendations, user behavior, and product filtering, with CORS support for web integration.
Features

Personalized Recommendations: Suggests top-5 products using cosine similarity on product embeddings, based on user purchases and views.
Conversational Explanations: Generates ~100-word explanations (e.g., “It’ll fit right into your style!”) using TinyLLaMA or Mistral, highlighting user preferences, ratings, and positive sentiments.
Sentiment Analysis: Employs distilbert for review sentiment, with a keyword-based fallback (positive: “great,” “cozy”; negative: “poor,” “terrible”) due to API issues.
FastAPI Endpoints:
GET /: Health check with system status.
GET /health: Alias for health check.
GET /users/{user_id}/behavior: User behavior (purchases, views, categories, ratings).
POST /recommendations: Personalized recommendations with explanations.
GET /products: Filter products by category, rating, or limit.


Database: ecommerce.db (~100 interactions, ~50 products, ~30 users).
Metrics: Tracks explanation quality (length ~100 words, Jaccard overlap ~0.4-0.5, sentiment score ~0.9).

Project Evolution
Initial Development

Built generate_recom.py using Flan-T5-Base for explanations, all-MiniLM-L6-v2 for embeddings, and SQLite for data.
Initially generated top-10 recommendations, later refined to top-5.
Early explanations were repetitive (e.g., “Sweater (Clothing): ...”) and short (e.g., 49 words, Jaccard 0.23).
Used keyword-based sentiment analysis due to distilbert 401 errors.

Improvements

Prompt Refinement: Created conversational prompts (“Hey, I think this is perfect for you!”) to target ~100 words and avoid repetitive product lists.
Humanization: Added humanize_explanation to remove technical terms (e.g., “POSITIVE, score: 0.75”) and repetitive lists, ensuring a natural tone.
LLM Switch: Replaced Flan-T5-Base with TinyLLaMA (1.1B) (2GB) for better natural language, as Mistral (7B) (14GB) was too large for smaller setups.
Sentiment: Expanded keyword lists (e.g., “cozy,” “uncomfortable”) for robust fallback.
HUGGINGFACE_TOKEN: Fixed lookup from hf_LvOqnLTW... to HUGGINGFACE_TOKEN.

FastAPI Integration

Developed a FastAPI app for scalable API deployment, using Mistral (7B) via Ollama for explanations.
Added CORS middleware, Pydantic models (RecommendationRequest, ProductRecommendation), and robust endpoints.
Improved error handling (e.g., HTTP 404 for invalid users, 500 for server errors).

Installation
Prerequisites

Python 3.8+
SQLite database (ecommerce.db) with products, users, and interactions tables.
Ollama server (http://localhost:11434) with Mistral 7B model.
Optional: GPU with ~4GB VRAM (TinyLLaMA) or ~14GB (Mistral).

Setup

Clone the repository:git clone <repository-url>
cd ecommerce-recommendation


Install dependencies:pip install fastapi uvicorn pandas numpy sentence-transformers torch transformers scikit-learn requests nltk
python -m nltk.downloader punkt punkt_tab


Set Hugging Face token:export HUGGINGFACE_TOKEN='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'  # Linux/Mac
set HUGGINGFACE_TOKEN=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    # Windows


Set up ecommerce.db:python init_db.py
python generate_data.py


Start Ollama server:ollama run mistral:7b-instruct-q4_K_M



Usage

Start the FastAPI server:uvicorn main:app --host 0.0.0.0 --port 1234


Access endpoints:
Swagger UI: http://localhost:1234/docs
Health Check: curl http://localhost:1234/
Recommendations:curl -X POST http://localhost:1234/recommendations -H "Content-Type: application/json" -d '{"user_id": 1, "top_k": 5, "min_rating": 4.0}'


User Behavior: curl http://localhost:1234/users/1/behavior
Products: curl http://localhost:1234/products?category=Clothing&min_rating=4.0&limit=10



Example Output
For userId=1, productId=19 (Sweater):
{
  "productId": 19,
  "productName": "Sweater",
  "productCategory": "Clothing",
  "productDescription": "Comfortable Sweater with premium fabric",
  "avgRating": 4.5,
  "explanation": "Hey, I think this Sweater is perfect for you! You’ve been eyeing comfy clothes like sweaters and denim trousers, checking them out multiple times. You’ve bought similar items, giving them high ratings around 4.2, and left reviews like ‘Love the cozy fit!’ Your style leans toward quality, versatile pieces, and this sweater’s premium fabric fits that vibe perfectly. It’s ideal for casual days or layering up. It’s just the kind of thing you love, and it’ll fit right into your style!"
}

Metrics: Length = 102, Jaccard Overlap = 0.46, Sentiment Score = 0.93
Current State

Strengths:
Delivers top-5 recommendations with ~100-word, conversational explanations.
Handles Clothing-heavy recommendations based on ecommerce.db.
Robust sentiment analysis with keyword fallback.
Scalable FastAPI app with clear endpoints and error handling.


Limitations:
Clothing-heavy recommendations due to biased ecommerce.db data.
Keyword-based sentiment is simple; distilbert requires a valid Hugging Face token.
Mistral (7B) needs ~14GB VRAM, less suitable for smaller setups.



Future Improvements

Diversify recommendations by updating generate_data.py to balance categories.
Resolve distilbert 401 errors or use a local sentiment model (e.g., bert-base-uncased).
Transition to Flask API (app.py) for compatibility.
Fine-tune Mistral prompts or test smaller LLMs (e.g., Phi-2) for low-memory setups.

Troubleshooting

Repetitive Explanations:
Adjust no_repeat_ngram_size or refine humanize_explanation regex:explanation = re.sub(r'(Sweater|Jacket).*?(,|$)', '', explanation)




Short Explanations:
Increase length_penalty (e.g., 2.0) or num_predict (e.g., 150) in Ollama payload.


Low Jaccard Overlap:
Check user behavior:from main import get_user_behavior
print(get_user_behavior(1))




Database Issues:
Verify ecommerce.db:conn = sqlite3.connect('ecommerce.db')
print(pd.read_sql('SELECT * FROM interactions WHERE userId = 1', conn))
conn.close()




Ollama Errors:
Ensure Ollama is running: ollama run mistral:7b-instruct-q4_K_M.


TensorFlow Warnings:
Run:export TF_ENABLE_ONEDNN_OPTS=0  # Linux/Mac
set TF_ENABLE_ONEDNN_OPTS=0    # Windows




MIT License
