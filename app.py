import streamlit as st
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the title of the Streamlit app
st.title("Movie Recommendation Chatbot")

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"  # Make sure this matches your FastAPI server port

# Input text box for user query about movies
query = st.text_input("Anything about movies")

# Button to trigger the movie recommendation request
if st.button("Filming it!"):
    try:
        # Send a POST request to the FastAPI server with the user query
        response = requests.post(f"{BASE_URL}/movies/", json={"query": query})
        # Raise an error for bad HTTP status codes
        response.raise_for_status()
        
        if response.status_code == 200:
            # Parse the JSON response to get the list of movies
            movies = response.json().get("movies")
            st.write("Movies:")
            # Display each movie's title, genre, and plot
            for movie in movies:
                st.write(f"* {movie['title']} ({movie['genre']}) - {movie['plot']}")
        else:
            # Log an error message if the request was unsuccessful
            logger.error(f"Failed to fetch movies. Status code: {response.status_code}, Response: {response.text}")
            st.error("Failed to fetch movies. Please try again.")
    except requests.exceptions.RequestException as e:
        # Log a request exception error message
        logger.error(f"RequestException: {e}")
        st.error("Unable to connect to the movie recommendation service. Please try again later.")
    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please try again later.")

# To run the Streamlit app, use the command: streamlit run app.py
