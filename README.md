# Movie Recommendation System with Metrics Evaluation and Fine Tuning

This project is a movie recommendation system that retrieves relevant movie data based on user queries and evaluates the performance of the recommendations using various metrics. The system leverages ChromaDB for storing and querying movie data and utilizes pre-trained language models for natural language processing tasks.

# youtube link.
[https://youtu.be/nvjtSLknYCs](https://www.youtube.com/watch?v=53bpGNaD5mk)

# Group Project
Rajan Panchal - 002626533 - https://www.linkedin.com/in/randomrajan/  
Saahil Karnik - 002209455- https://www.linkedin.com/in/saahil-karnik-84a029165/   
Niruthiha Selvanayagam - 002413183 - https://www.linkedin.com/in/niruthi-selva/?originalSubdomain=ca

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

### Run the metrics file
python metrics.py

### FineTuning
python finetune.py

### start the frontend
streamlit run app.py

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


## Explainging Code file.

Here’s a brief explanation of each of the code files you provided:

### app.py:
This file is likely the main application script for deploying the movie recommendation system. It probably contains the code to start the server, handle HTTP requests, and interact with the recommendation model to serve movie suggestions and information to users. It could be using a web framework like Flask or FastAPI to handle these interactions.

### finetune.py:
This script is responsible for fine-tuning the recommendation model. Fine-tuning involves further training a pre-existing model on additional data or tweaking it to improve its accuracy. This script likely loads the base model, prepares additional data, and adjusts the model's parameters to better fit the new data or target audience.

### main.py:
The main.py file appears to be the entry point for training the recommendation model from scratch or from a pre-trained state. It likely includes data loading, preprocessing, model definition, and the training loop. This script might be where the core machine learning or deep learning algorithm is implemented.

### metrics.py:
The metrics.py script is used for evaluating the performance of the recommendation model. It likely calculates various metrics such as accuracy, precision, recall, or others relevant to recommendation systems. The script might compare the model’s predictions against a validation or test dataset to assess how well the model is performing.

### movie_data.json:
This JSON file contains the dataset of movies, including details like titles, genres, plots, release dates, ratings, and cast members. This dataset is utilized by the model for making recommendations and is also displayed to users as part of the application’s information retrieval feature.

## Contributing  
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.  

## License  
This project is licensed under the MIT License.  
)
