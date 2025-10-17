E-Commerce Recommendation System
Overview
This project is an AI-powered e-commerce recommendation system built with FastAPI, delivering personalized product recommendations with human-like explanations. It uses user behavior (views, purchases, reviews) from ecommerce.db to recommend products, leveraging embeddings (all-MiniLM-L6-v2) and LLMs (Flan-T5-Base, TinyLLaMA, Mistral) for explanations. The system targets top-5 recommendations, conversational explanations (~100 words), and handles sentiment analysis with distilbert or a keyword-based fallback. The FastAPI app provides endpoints for recommendations, user behavior, and product filtering, with CORS support for web integration.
Features

Recommendation Engine: Uses cosine similarity on product embeddings to suggest top-5 products based on user interactions (purchases, views).
Human-Like Explanations: Generates conversational explanations (e.g., “Hey, this Sweater is perfect for you!”) using LLMs, highlighting user style, ratings, and positive sentiments.
Sentiment Analysis: Applies distilbert for review sentiment, with a keyword-based fallback (e.g., positive: “great,” “cozy”; negative: “poor,” “terrible”) due to API authentication issues.
FastAPI Endpoints:
GET /: Health check with system status.
GET /health: Alias for health check.
GET /users/{user_id}/behavior: Returns user behavior (purchases, views, categories, ratings).
POST /recommendations: Generates personalized recommendations with explanations.
GET /products: Filters products by category, rating, or limit.


Database: Uses ecommerce.db (~100 interactions, ~50 products, ~30 users) for user and product data.
Metrics: Tracks explanation quality via length (100 words), Jaccard overlap (0.4-0.5), and sentiment score (~0.9).

Project Evolution

Initial Script (generate_recom.py):

Used Flan-T5-Base for explanations, all-MiniLM-L6-v2 for embeddings, and SQLite for data.
Generated top-10 recommendations, later refined to top-5 per user request.
Explanations were list-heavy (e.g., “Sweater (Clothing): ...”) and short (e.g., 49 words).
Sentiment analysis relied on keyword-based fallback due to distilbert 401 errors.


Improvements:

Prompt Refinement: Updated prompts to be conversational (“Hey, I think this is perfect for you!”), targeting ~100 words and avoiding repetitive product lists.
Humanization Step: Added humanize_explanation to remove technical terms (e.g., “POSITIVE, score: 0.75”) and repetitive lists, ensuring natural tone.
LLM Switch: Replaced Flan-T5-Base with TinyLLaMA (1.1B) for better natural language, as Mistral (7B) was too large (14GB). TinyLLaMA (2GB) fits 4GB GPU/CPU.
Sentiment Fix: Expanded keyword lists (e.g., added “cozy,” “uncomfortable”) for robust fallback sentiment analysis.
HUGGINGFACE_TOKEN: Fixed environment variable lookup from hf_LvOqnLTW... to HUGGINGFACE_TOKEN.


FastAPI Integration:

Built a FastAPI app for scalable API deployment, replacing script-based testing.
Uses Mistral (7B, via Ollama) for explanations, with improved prompt for concise, personalized outputs.
Added CORS middleware, Pydantic models (RecommendationRequest, ProductRecommendation), and endpoints for recommendations and behavior analysis.
Handles errors gracefully (e.g., HTTP 404 for invalid users, 500 for server errors).



Setup Instructions
Prerequisites

Python 3.8+
SQLite database (ecommerce.db) with products, users, and interactions tables.
Ollama server running locally (http://localhost:11434) with Mistral 7B model.
Optional: GPU with ~4GB VRAM for TinyLLaMA or ~14GB for Mistral.

Installation

Clone the repository:git clone <repository-url>
cd ecommerce-recommendation


Install dependencies:pip install fastapi uvicorn pandas numpy sentence-transformers torch transformers scikit-learn requests nltk
python -m nltk.downloader punkt punkt_tab


Set up Hugging Face token:export HUGGINGFACE_TOKEN='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'  # Linux/Mac
set HUGGINGFACE_TOKEN=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    # Windows


Create and populate ecommerce.db (example script: init_db.py):python init_db.py
python generate_data.py


Start Ollama server:ollama run mistral:7b-instruct-q4_K_M



Running the API

Start the FastAPI server:uvicorn main:app --host 0.0.0.0 --port 1234


Access the API:
Swagger UI: http://localhost:1234/docs
Health check: curl http://localhost:1234/
Recommendations: curl -X POST http://localhost:1234/recommendations -H "Content-Type: application/json" -d '{"user_id": 1, "top_k": 5, "min_rating": 4.0}'
User behavior: curl http://localhost:1234/users/1/behavior



Example Output
For userId=1, productId=19 (Sweater):
Explanation for User 1, Product 19 (Sweater):
Hey, I think this Sweater is perfect for you! You’ve been eyeing comfy clothes like sweaters and denim trousers, checking them out multiple times. You’ve bought similar items, giving them high ratings around 4.2, and left reviews like “Love the cozy fit!” Your style leans toward quality, versatile pieces, and this sweater’s premium fabric, rated 4.5, fits that vibe perfectly. It’s ideal for casual days or layering up. It’s just the kind of thing you love, and it’ll fit right into your style!
Metrics: Length = 102, Jaccard Overlap = 0.46, Sentiment Score = 0.93

Current State

Strengths:
Delivers top-5 recommendations with ~100-word, conversational explanations.
Handles Clothing-heavy recommendations (e.g., Sweater, Jacket) based on user behavior in ecommerce.db.
Robust sentiment analysis with keyword fallback.
Scalable FastAPI app with clear endpoints and error handling.


Limitations:
Recommendations are Clothing-heavy, likely due to biased ecommerce.db data.
Keyword-based sentiment analysis is simple; distilbert requires a valid Hugging Face token.
Mistral (7B) requires ~14GB VRAM, which may not suit smaller setups (TinyLLaMA used for testing).



Future Steps

Diversify Recommendations: Modify generate_data.py to balance categories in ecommerce.db.
Improve Sentiment: Resolve distilbert 401 errors with a valid token or integrate a local sentiment model.
Flask API: Transition to Flask for compatibility with existing systems (planned as Step 5, app.py).
Optimize Explanations: Fine-tune Mistral prompts or switch to smaller LLMs (e.g., Phi-2) for low-memory setups.

Troubleshooting

Repetitive Explanations: Adjust no_repeat_ngram_size in generate_explanation or refine humanize_explanation regex.
Short Explanations: Increase length_penalty (e.g., 2.0) or num_predict in Ollama payload.
Low Jaccard: Verify context:from fastapi_app import retrieve_context
print(retrieve_context(1, 19))


Database Issues:conn = sqlite3.connect('ecommerce.db')
print(pd.read_sql('SELECT * FROM interactions WHERE userId = 1', conn))
conn.close()


Ollama Errors: Ensure Ollama server is running (ollama run mistral:7b-instruct-q4_K_M).
TensorFlow Warnings:set TF_ENABLE_ONEDNN_OPTS=0



Contributors

You: Designed the recommendation system, implemented generate_recom.py with Flan-T5-Base and TinyLLaMA, refined prompts for human-like explanations, addressed repetitive outputs, and integrated FastAPI with Mistral via Ollama.

License
MIT License
