import streamlit as st
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Movie Recommendation Chatbot")

BASE_URL = "http://localhost:8000"  # Make sure this matches your FastAPI server port

query = st.text_input("Anything about movies")

if st.button("Filming it!"):
    try:
        response = requests.post(f"{BASE_URL}/movies/", json={"query": query})
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        if response.status_code == 200:
            movies = response.json().get("movies")
            st.write("Movies:")
            for movie in movies:
                st.write(f"* {movie['title']} ({movie['genre']}) - {movie['plot']}")
        else:
            logger.error(f"Failed to fetch movies. Status code: {response.status_code}, Response: {response.text}")
            st.error("Failed to fetch movies. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException: {e}")
        st.error("Unable to connect to the movie recommendation service. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please try again later.")

# Run the Streamlit app with: streamlit run app.py
