import os
from fastapi import FastAPI, HTTPException  # FastAPI for building the API, HTTPException for error handling
from pydantic import BaseModel  # Pydantic for data validation and serialization
from langchain import OpenAI  # OpenAI integration via LangChain
from chromadb import Client  # ChromaDB client for vector database management
from chromadb.config import Settings  # Settings configuration for ChromaDB
import logging  # Logging for monitoring and debugging
from dotenv import load_dotenv  # Load environment variables from a .env file
import json  # JSON handling for reading movie data
from metrics import (  # Importing custom metrics functions for evaluating recommendations
    retrieve_context, generate_answer, extract_entities,
    context_precision, context_recall, context_relevance,
    context_entity_recall, noise_robustness, faithfulness,
    answer_relevance, information_integration, counterfactual_robustness,
    negative_rejection, measure_latency, format_recommendations, align_lengths
)
from sentence_transformers import SentenceTransformer  # HuggingFace's SentenceTransformer for embedding text

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
    # Initialize GPT-4 model from OpenAI and SentenceTransformer embeddings
    llm = OpenAI(api_key=openai_api_key, model_name="gpt-4")
    embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    logger.info("Successfully initialized LLM and embeddings.")
except Exception as e:
    logger.error(f"Error initializing LLM or embeddings: {e}")
    raise

# Initialize ChromaDB client and create a collection
try:
    client = Client(Settings(persist_directory="chroma_db"))  # Setting up ChromaDB with a persistent directory
    collection = client.create_collection("movies")  # Creating a collection in ChromaDB to store movie data
    logger.info("Successfully initialized ChromaDB.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise

# Define the path to your JSON file containing movie data
json_file_path = 'movie_data.json'

# Open and read the JSON file
with open(json_file_path, 'r') as file:
    movie_data = json.load(file)

# Add movie data to ChromaDB collection
try:
    for idx, movie in enumerate(movie_data):  # Iterate through each movie in the dataset
        # Combine plot and genre for embedding
        text_to_embed = f"{movie['plot']} [GENRE: {movie['genre']}]"
        plot_embedding = embeddings.encode([text_to_embed])[0]  # Get the first (and only) embedding
        logger.info(f"Type before conversion: {type(plot_embedding)}")  # Log the type before conversion

        if isinstance(plot_embedding, list):
            logger.info("Embedding is already a list.")
        else:
            logger.info("Converting ndarray to list.")
            plot_embedding = plot_embedding.tolist()  # Convert the embedding to a list if necessary

        logger.info(f"Type after conversion: {type(plot_embedding)}")  # Log the type after conversion
        logger.info(f"Embedding content: {plot_embedding}")  # Log the embedding content for debugging

        # Convert the 'cast' list to a comma-separated string for storage
        movie['cast'] = ', '.join(movie['cast'])

        # Add movie metadata and embeddings to the ChromaDB collection
        collection.add(ids=[str(idx)], embeddings=[plot_embedding], metadatas=[movie], documents=[text_to_embed])
    logger.info("Successfully added movie data to ChromaDB.")
except Exception as e:
    logger.error(f"Error adding movie data to ChromaDB: {e}")
    raise

# Pydantic model for user query
class UserQuery(BaseModel):
    query: str = None  # User query input model, where query is an optional string

# Dummy ground truth and reference data for testing purposes
relevant_contexts = [  # Replace with actual ground truth in production
    "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with Emma Thomas.",
    "The Godfather is a 1972 American crime film directed by Francis Ford Coppola who co-wrote the screenplay with Mario Puzo.",
    "Pulp Fiction is a 1994 American crime film written and directed by Quentin Tarantino, who conceived it with Roger Avary.",
    "The Dark Knight is a 2008 superhero film directed, co-produced, and co-written by Christopher Nolan."
]

reference_answers = [  # Sample reference answers for testing
    "Inception is a science fiction action film directed by Christopher Nolan.",
    "The Godfather is a crime film directed by Francis Ford Coppola.",
    "Pulp Fiction is a crime film written and directed by Quentin Tarantino.",
    "The Dark Knight is a superhero film directed by Christopher Nolan."
]

counterfactual_queries = [  # Test queries that shouldn't yield positive results
    "Who directed The Godfather in 2020?",
    "What is the plot of Pulp Fiction directed by Steven Spielberg?",
    "In which year was Inception released by Martin Scorsese?",
    "Who starred in The Dark Knight directed by Alfred Hitchcock?"
]

negative_queries = [  # Queries about fictional movies for robustness testing
    "Tell me about the plot of a non-existent movie.",
    "What is the release date of the imaginary sequel to Inception?",
    "Who are the actors in the made-up film 'Random Movie 2025'?",
    "Describe the storyline of a fictional movie never produced."
]

# API endpoint to handle movie recommendations
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
        if user_query.query:  # If a query is provided
            logger.info(f"Received query: {user_query.query}")

            # Retrieve and measure latency for recommendations
            latency, recommendations = measure_latency(retrieve_context, collection, user_query.query)
            logger.info(f"Query results: {recommendations}")

            if not recommendations:  # If no recommendations found
                logger.info("No recommendations found.")
                raise HTTPException(status_code=404, detail="No recommendations found")

            # Ensure relevant_contexts has the same number of elements as recommendations
            relevant_contexts_extended, _ = align_lengths(relevant_contexts, recommendations)

            # Calculate various performance metrics
            precision = context_precision(recommendations, relevant_contexts_extended)
            recall = context_recall(recommendations, relevant_contexts_extended)
            relevance = context_relevance(recommendations, user_query.query)
            entity_recall = context_entity_recall(recommendations, {"Christopher Nolan", "Francis Ford Coppola", "Quentin Tarantino"})  # Placeholder
            noise_robustness_score = noise_robustness(recommendations, ["noisy query 1", "noisy query 2"])  # Placeholder
            generated_answers = [generate_answer(recommendations, user_query.query)]
            faithfulness_score = faithfulness(generated_answers, reference_answers)  # Placeholder
            answer_relevance_score = answer_relevance(generated_answers, user_query.query)
            information_integration_score = information_integration(generated_answers, reference_answers)  # Placeholder
            counterfactual_robustness_score = counterfactual_robustness(generated_answers, counterfactual_queries)  # Placeholder
            negative_rejection_score = negative_rejection(generated_answers, negative_queries)  # Placeholder

            # Compile metrics into a dictionary
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

            return {"movies": recommendations[:5], "metrics": metrics}  # Return top 5 recommendations and metrics
        else:  # If no query is provided, return all movies
            results = collection.get()
            logger.info(f"All movies: {results}")
            all_movies = results.get("metadatas", [])
            return {"movies": all_movies}
    except Exception as e:
        logger.error(f"Error during movie retrieval: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the FastAPI application using Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the app on all available IP addresses on port 8000
