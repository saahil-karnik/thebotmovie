# import os
# import requests
# import json

# # Ensure you have set your TMDB_API_KEY environment variable
# tmdb_api_key = os.getenv("TMDB_API_KEY")
# if not tmdb_api_key:
#     raise ValueError("Please set the TMDB_API_KEY environment variable.")

# def fetch_movie_data(tmdb_api_key, page):
#     url = f"https://api.themoviedb.org/3/movie/popular?api_key={tmdb_api_key}&language=en-US&page={page}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json().get("results", [])
#     else:
#         raise Exception(f"Failed to fetch data from TMDB: {response.status_code}")

# def get_genre_name(genre_id, genres):
#     for genre in genres:
#         if genre['id'] == genre_id:
#             return genre['name']
#     return "Unknown"

# def fetch_genres(tmdb_api_key):
#     url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={tmdb_api_key}&language=en-US"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json().get("genres", [])
#     else:
#         raise Exception(f"Failed to fetch genres from TMDB: {response.status_code}")

# def fetch_movie_details(tmdb_api_key, movie_id):
#     url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US&append_to_response=credits,videos"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise Exception(f"Failed to fetch movie details from TMDB: {response.status_code}")

# genres = fetch_genres(tmdb_api_key)
# movie_data = []

# for page in range(1, 11):  # Fetch multiple pages to get up to 200 movies
#     movies = fetch_movie_data(tmdb_api_key, page)
#     for movie in movies:
#         movie_id = movie['id']
#         details = fetch_movie_details(tmdb_api_key, movie_id)

#         genre_names = [get_genre_name(genre_id, genres) for genre_id in movie['genre_ids']]
#         genre = genre_names[0] if genre_names else "Unknown"

#         cast = [member['name'] for member in details['credits']['cast'][:5]]  # Get top 5 cast members
#         trailers = [video['key'] for video in details['videos']['results'] if video['type'] == 'Trailer']
#         trailer_link = f"https://www.youtube.com/watch?v={trailers[0]}" if trailers else "N/A"

#         movie_entry = {
#             "title": movie['title'],
#             "genre": genre,
#             "plot": movie['overview'],
#             "release_date": movie.get('release_date', 'N/A'),
#             "vote_average": movie.get('vote_average', 'N/A'),
#             "vote_count": movie.get('vote_count', 'N/A'),
#             "popularity": movie.get('popularity', 'N/A'),
#             "cast": cast,
#             "trailer_link": trailer_link
#         }
#         movie_data.append(movie_entry)
#         if len(movie_data) >= 500:
#             break
#     if len(movie_data) >= 500:
#         break

# # Save the data to a file
# with open('movie_data.json', 'w') as f:
#     json.dump(movie_data, f, indent=4)

# # Print the number of movies retrieved
# print(f"Total movies retrieved: {len(movie_data)}")

import os
import requests
import json
import time

# Ensure you have set your TMDB_API_KEY environment variable
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

def fetch_movie_details(tmdb_api_key, movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US&append_to_response=credits,videos"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch movie details from TMDB: {response.status_code}")

genres = fetch_genres(tmdb_api_key)
movie_data = []

for page in range(1, 101):  # Fetch 100 pages to get up to 2000 movies
    movies = fetch_movie_data(tmdb_api_key, page)
    for movie in movies:
        movie_id = movie['id']
        details = fetch_movie_details(tmdb_api_key, movie_id)
        
        genre_names = [get_genre_name(genre_id, genres) for genre_id in movie['genre_ids']]
        genre = genre_names[0] if genre_names else "Unknown"

        cast = [member['name'] for member in details['credits']['cast'][:5]]  # Get top 5 cast members
        trailers = [video['key'] for video in details['videos']['results'] if video['type'] == 'Trailer']
        trailer_link = f"https://www.youtube.com/watch?v={trailers[0]}" if trailers else "N/A"

        movie_entry = {
            "title": movie['title'],
            "genre": genre,
            "plot": movie['overview'],
            "release_date": movie.get('release_date', 'N/A'),
            "vote_average": movie.get('vote_average', 'N/A'),
            "vote_count": movie.get('vote_count', 'N/A'),
            "popularity": movie.get('popularity', 'N/A'),
            "cast": cast,
            "trailer_link": trailer_link
        }
        movie_data.append(movie_entry)
        if len(movie_data) >= 1000:
            break
    if len(movie_data) >= 1000:
        break
    # Sleep for a short duration to avoid hitting the API rate limit
    time.sleep(0.5)

# Save the data to a file
with open('movie_data.json', 'w') as f:
    json.dump(movie_data, f, indent=4)

# Print the number of movies retrieved
print(f"Total movies retrieved: {len(movie_data)}")
