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
query = st.text_input("Ask anything about movies")

# Button to trigger the movie recommendation request
if st.button("Filming it!"):
    try:
        # Send a POST request to the FastAPI server with the user query
        response = requests.post(f"{BASE_URL}/movies/", json={"query": query})
        # Raise an error for bad HTTP status codes
        response.raise_for_status()

        if response.status_code == 200:
            # Parse the JSON response to get the list of movies and metrics
            response_json = response.json()
            movies = response_json.get("movies")
            metrics = response_json.get("metrics")

            st.write("Movies:")
            # Display each movie's details
            for movie in movies:
                st.write(f"**Title:** {movie['title']}")
                st.write(f"**Genre:** {movie['genre']}")
                st.write(f"**Plot:** {movie['plot']}")
                st.write(f"**Release Date:** {movie['release_date']}")
                st.write(f"**Vote Average:** {movie['vote_average']}")
                st.write(f"**Vote Count:** {movie['vote_count']}")
                st.write(f"**Popularity:** {movie['popularity']}")
                st.write(f"**Cast:** {movie['cast']}")
                st.write(f"**Trailer Link:** [Watch Trailer]({movie['trailer_link']})")
                st.write("-" * 40)

            st.write("Metrics:")
            # Display metrics
            # for metric, value in metrics.items():
            #     st.write(f"{metric}: {value}")

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
