# Movie Recommendation System with Metrics Evaluation

This project is a movie recommendation system that retrieves relevant movie data based on user queries and evaluates the performance of the recommendations using various metrics. The system leverages ChromaDB for storing and querying movie data and utilizes pre-trained language models for natural language processing tasks.

## Introduction

This project provides a REST API for retrieving movie recommendations based on user queries. It also evaluates the performance of the recommendations using metrics such as context precision, context recall, context relevance, and more. The project uses FastAPI for the API, ChromaDB for database management, and various NLP models for processing queries and calculating metrics.

## Features

- Retrieve movie recommendations based on user queries.
- Evaluate recommendation performance using a range of metrics.
- Use pre-trained language models for natural language processing tasks.
- Store and query movie data using ChromaDB.

## Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

5. Ensure you have a file named `movie_data.json` with your movie data in the root directory.

## Usage

### Running the API

To start the FastAPI server, run:

uvicorn main:app --reload
The API will be accessible at http://127.0.0.1:8000.

API Endpoints
POST /movies/: Retrieve movie recommendations based on a user query.
Example Request
{
    "query": "Tell me about movies with strong female leads."
}
Example Response
{
    "movies": [
        {
            "title": "Mad Max: Fury Road",
            "genre": "Action",
            "plot": "In a post-apocalyptic wasteland, Max teams up with Furiosa to escape a tyrant who controls the land's water supply.",
            "cast": "Tom Hardy, Charlize Theron, Nicholas Hoult",
            "release_date": "2015-05-15",
            "popularity": 100.0,
            "vote_average": 7.8
        },
        // More movies...
    ],
    "metrics": {
        "Context Precision": 0.28,
        "Context Recall": 0.28,
        "Context Relevance": 0.048,
        "Context Entity Recall": 0.0,
        "Noise Robustness": 1.5,
        "Faithfulness": 1.0,
        "Answer Relevance": 0.126,
        "Information Integration": 1.0,
        "Counterfactual Robustness": 1.0,
        "Negative Rejection": 0.75,
        "Latency": 0.574
    }
}
Running Metrics Evaluation
To run the metrics evaluation script, use:

python metrics.py

## Metrics
The system evaluates the performance of movie recommendations using the following metrics:

Context Precision: Measures the accuracy of the retrieved context.

Context Recall: Measures the ability to retrieve all relevant contexts.

-Context Relevance: Assesses the relevance of the retrieved context to the query.
-Context Entity Recall: Determines the ability to recall relevant entities within the context.
-Noise Robustness: Tests the system's ability to handle noisy or irrelevant inputs.
-Faithfulness: Measures the accuracy and reliability of the generated answers.
-Answer Relevance: Evaluates the relevance of the generated answers to the query.
-Information Integration: Assesses the ability to integrate and present information cohesively.
-Counterfactual Robustness: Tests the robustness of the system against counterfactual or contradictory queries.
-Negative Rejection: Measures the system's ability to reject and handle negative or inappropriate queries.
-Latency: Measures the response time from receiving a query to delivering an answer.

## Improvement Tips
-Enhance Data Quality: Ensure that the movie data in movie_data.json is accurate and comprehensive.
-Improve Entity Extraction: Fine-tune the Named Entity Recognition (NER) model to better identify entities within the movie data.
-Optimize Retrieval Algorithms: Experiment with different retrieval algorithms or embeddings to improve precision and recall.
-Increase Model Performance: Use more advanced language models and fine-tune them on a specific movie dataset.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

# youtube link.
https://youtu.be/nvjtSLknYCs
