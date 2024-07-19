import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAI
from chromadb import Client
from chromadb.config import Settings
import logging
from dotenv import load_dotenv
import json
from metrics import (
    retrieve_context, generate_answer, extract_entities,  # Add extract_entities import
    context_precision, context_recall, context_relevance,
    context_entity_recall, noise_robustness, faithfulness,
    answer_relevance, information_integration, counterfactual_robustness,
    negative_rejection, measure_latency
)
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the OpenAI language model and HuggingFace embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("Please set the OPENAI_API_KEY environment variable.")
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

try:
    llm = OpenAI(api_key=openai_api_key, model_name="text-davinci-003")
    embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("Successfully initialized LLM and embeddings.")
except Exception as e:
    logger.error(f"Error initializing LLM or embeddings: {e}")
    raise

# Initialize ChromaDB client and create a collection
try:
    client = Client(Settings(persist_directory="chroma_db"))
    collection = client.create_collection("movies")
    logger.info("Successfully initialized ChromaDB.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise

# Define the path to your JSON file
json_file_path = 'movie_data.json'

# Open and read the JSON file
with open(json_file_path, 'r') as file:
    movie_data = json.load(file)

# Add movie data to ChromaDB collection
try:
    for idx, movie in enumerate(movie_data):
        text_to_embed = f"{movie['plot']} [GENRE: {movie['genre']}]"
        plot_embedding = embeddings.encode([text_to_embed])[0]  # Get the first (and only) embedding
        logger.info(f"Type before conversion: {type(plot_embedding)}")  # Log the type before conversion

        if isinstance(plot_embedding, list):
            logger.info("Embedding is already a list.")
        else:
            logger.info("Converting ndarray to list.")
            plot_embedding = plot_embedding.tolist()  # Convert the embedding to a list

        logger.info(f"Type after conversion: {type(plot_embedding)}")  # Log the type after conversion
        logger.info(f"Embedding content: {plot_embedding}")  # Log the embedding content for debugging

        collection.add(ids=[str(idx)], embeddings=[plot_embedding], metadatas=[movie], documents=[text_to_embed])
    logger.info("Successfully added movie data to ChromaDB.")
except Exception as e:
    logger.error(f"Error adding movie data to ChromaDB: {e}")
    raise

# Pydantic model for user query
class UserQuery(BaseModel):
    query: str = None

# Dummy ground truth and reference data for testing purposes
relevant_contexts = ["Sample context 1 for query", "Sample context 2 for query"]  # Replace with actual ground truth
reference_answers = ["Expected answer 1", "Expected answer 2"]
counterfactual_queries = ["Counterfactual query 1", "Counterfactual query 2"]
negative_queries = ["Negative query 1", "Negative query 2"]

# FastAPI route to handle movie recommendations
@app.post("/movies/")
async def get_movies(user_query: UserQuery):
    """
    Handle POST requests to retrieve movie recommendations based on user query.
    If a query is provided, it retrieves the most relevant movies from ChromaDB.
    If no query is provided, it retrieves all movies from ChromaDB.

    Parameters:
    - user_query: UserQuery - The user's query for movie recommendations.

    Returns:
    - A dictionary containing the recommended movies.
    """
    try:
        if user_query.query:
            logger.info(f"Received query: {user_query.query}")

            latency, recommendations = measure_latency(retrieve_context, collection, user_query.query)
            logger.info(f"Query results: {recommendations}")

            if not recommendations:
                logger.info("No recommendations found.")
                raise HTTPException(status_code=404, detail="No recommendations found")

            # Calculate metrics
            precision = context_precision(recommendations, relevant_contexts)
            recall = context_recall(recommendations, relevant_contexts)
            relevance = context_relevance(recommendations, user_query.query)
            entity_recall = context_entity_recall(recommendations, {"entity1", "entity2"})  # Placeholder
            noise_robustness_score = noise_robustness(recommendations, ["noisy query 1", "noisy query 2"])  # Placeholder
            generated_answers = [generate_answer(recommendations, user_query.query)]
            faithfulness_score = faithfulness(generated_answers, reference_answers)  # Placeholder
            answer_relevance_score = answer_relevance(generated_answers, user_query.query)
            information_integration_score = information_integration(generated_answers, reference_answers)  # Placeholder
            counterfactual_robustness_score = counterfactual_robustness(generated_answers, counterfactual_queries)  # Placeholder
            negative_rejection_score = negative_rejection(generated_answers, negative_queries)  # Placeholder

            metrics = {
                "Context Precision": precision,
                "Context Recall": recall,
                "Context Relevance": relevance,
                "Context Entity Recall": entity_recall,
                "Noise Robustness": noise_robustness_score,
                "Faithfulness": faithfulness_score,
                "Answer Relevance": answer_relevance_score,
                "Information Integration": information_integration_score,
                "Counterfactual Robustness": counterfactual_robustness_score,
                "Negative Rejection": negative_rejection_score,
                "Latency": latency,
            }

            return {"movies": recommendations[:5], "metrics": metrics}
        else:
            results = collection.get()
            logger.info(f"All movies: {results}")
            all_movies = results.get("metadatas", [])
            return {"movies": all_movies}
    except Exception as e:
        logger.error(f"Error during movie retrieval: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
