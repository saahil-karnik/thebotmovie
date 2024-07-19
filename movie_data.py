import os
import requests
import json

tmdb_api_key = os.getenv("TMDB_API_KEY")
if not tmdb_api_key:
    raise ValueError("Please set the TMDB_API_KEY environment variable.")

def fetch_movie_data(tmdb_api_key, page):
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={tmdb_api_key}&language=en-US&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        raise Exception(f"Failed to fetch data from TMDB: {response.status_code}")

def get_genre_name(genre_id, genres):
    for genre in genres:
        if genre['id'] == genre_id:
            return genre['name']
    return "Unknown"

def fetch_genres(tmdb_api_key):
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={tmdb_api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("genres", [])
    else:
        raise Exception(f"Failed to fetch genres from TMDB: {response.status_code}")

genres = fetch_genres(tmdb_api_key)
movie_data = []

for page in range(1, 11):  # Fetch multiple pages to get 200 movies
    movies = fetch_movie_data(tmdb_api_key, page)
    for movie in movies:
        genre_names = [get_genre_name(genre_id, genres) for genre_id in movie['genre_ids']]
        genre = genre_names[0] if genre_names else "Unknown"
        movie_entry = {
            "title": movie['title'],
            "genre": genre,
            "plot": movie['overview']
        }
        movie_data.append(movie_entry)
        if len(movie_data) >= 500:
            break
    if len(movie_data) >= 500:
        break

# Save the data to a file
with open('movie_data.json', 'w') as f:
    json.dump(movie_data, f, indent=4)

# Print the movie data
for movie in movie_data:
    print(movie)
