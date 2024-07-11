import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import Client
from chromadb.config import Settings
import logging
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the language model and vector database
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("Please set the OPENAI_API_KEY environment variable.")
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

try:
    llm = OpenAI(api_key=openai_api_key, model_name="text-davinci-003")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Successfully initialized LLM and embeddings.")
except Exception as e:
    logger.error(f"Error initializing LLM or embeddings: {e}")
    raise

# Initialize ChromaDB
try:
    client = Client(Settings(persist_directory="chroma_db"))
    collection = client.create_collection("movies")
    logger.info("Successfully initialized ChromaDB.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise

# Example movie data
movie_data = [
    {"title": "Inception", "genre": "Sci-Fi", "plot": "A thief who steals corporate secrets through the use of dream-sharing technology."},
    {"title": "The Matrix", "genre": "Action", "plot": "A computer hacker learns about the true nature of reality and his role in the war against its controllers."},
    {"title": "Inception", "genre": "Sci-Fi", "plot": "A thief who steals corporate secrets through the use of dream-sharing technology."},
    {"title": "The Matrix", "genre": "Sci-Fi", "plot": "A computer hacker learns about the true nature of reality and his role in the war against its controllers."},
    {"title": "Jurassic Park", "genre": "Sci-Fi", "plot": "A theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok."},
    {"title": "The Godfather", "genre": "Crime", "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
    {"title": "Pulp Fiction", "genre": "Crime", "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
]

# Add data to ChromaDB
try:
    for idx, movie in enumerate(movie_data):
        text_to_embed = f"{movie['plot']} [GENRE: {movie['genre']}]"
        plot_embedding = embeddings.embed_documents([text_to_embed])[0]
        collection.add(ids=[str(idx)], embeddings=[plot_embedding], metadatas=[movie], documents=[text_to_embed])
    logger.info("Successfully added movie data to ChromaDB.")
except Exception as e:
    logger.error(f"Error adding movie data to ChromaDB: {e}")
    raise

class UserQuery(BaseModel):
    query: str = None

@app.post("/movies/")
async def get_movies(user_query: UserQuery):
    try:
        if user_query.query:
            logger.info(f"Received query: {user_query.query}")
            query_embedding = embeddings.embed_documents([user_query.query])[0]
            logger.info(f"Query embedding: {query_embedding}")

            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            logger.info(f"Query results: {results}")

            recommendations = []
            for metadata_list in results.get('metadatas', []):
                recommendations.extend(metadata_list)

            if not recommendations:
                logger.info("No recommendations found.")
                raise HTTPException(status_code=404, detail="No recommendations found")

            return {"movies": recommendations[:2]}
        else:
            results = collection.get()
            logger.info(f"All movies: {results}")
            all_movies = results.get("metadatas", [])
            return {"movies": all_movies}
    except Exception as e:
        logger.error(f"Error during movie retrieval: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the app with: uvicorn main:app --reload
