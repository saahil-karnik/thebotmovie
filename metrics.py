import time
import numpy as np
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from evaluate import load
import logging
import re
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the ROUGE metric from Hugging Face
rouge = load("rouge")

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Load the NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Sample functions to retrieve context and generate answers
def retrieve_context(query, collection, n_results=3):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    contexts = [metadata for result in results['metadatas'] for metadata in result]
    return contexts

def generate_answer(contexts, query):
    answer = "Generated answer based on contexts and query"
    return answer

# Function to extract entities from a given context string
def extract_entities(context):
    entities = ner_pipeline(context)
    logger.info(f"Extracted entities: {entities}")
    entity_list = [entity['word'] for entity in entities if 'entity_group' in entity and entity['entity_group'] in ['PER', 'ORG', 'LOC']]
    return entity_list

def format_recommendations(recommendations):
    formatted = []
    for rec in recommendations:
        if isinstance(rec, dict):
            formatted.append(" ".join([f"{key}: {value}" for key, value in rec.items()]))
        else:
            formatted.append(rec)
    return formatted

def align_lengths(predictions, references):
    if len(predictions) > len(references):
        references.extend([''] * (len(predictions) - len(references)))
    elif len(predictions) < len(references):
        predictions.extend([''] * (len(references) - len(predictions)))
    return predictions, references

# Custom function to calculate context precision using ROUGE
def context_precision(retrieved_contexts, relevant_contexts):
    formatted_retrieved = format_recommendations(retrieved_contexts)
    formatted_retrieved, relevant_contexts = align_lengths(formatted_retrieved, relevant_contexts)
    logger.info(f"Formatted Retrieved Contexts: {formatted_retrieved}")
    logger.info(f"Relevant Contexts: {relevant_contexts}")
    scores = rouge.compute(predictions=formatted_retrieved, references=relevant_contexts)
    logger.info(f"ROUGE Scores for Precision: {scores}")
    precision = scores['rouge1']
    return precision

# Custom function to calculate context recall using ROUGE
def context_recall(retrieved_contexts, relevant_contexts):
    formatted_retrieved = format_recommendations(retrieved_contexts)
    formatted_retrieved, relevant_contexts = align_lengths(formatted_retrieved, relevant_contexts)
    logger.info(f"Formatted Retrieved Contexts: {formatted_retrieved}")
    logger.info(f"Relevant Contexts: {relevant_contexts}")
    scores = rouge.compute(predictions=formatted_retrieved, references=relevant_contexts)
    logger.info(f"ROUGE Scores for Recall: {scores}")
    recall = scores['rouge1']
    return recall

# Custom function to calculate context relevance using ROUGE
def context_relevance(retrieved_contexts, query):
    formatted_retrieved = format_recommendations(retrieved_contexts)
    scores = rouge.compute(predictions=formatted_retrieved, references=[query] * len(formatted_retrieved))
    logger.info(f"ROUGE Scores for Relevance: {scores}")
    relevance = scores['rouge1']
    return relevance

def context_entity_recall(retrieved_contexts, relevant_entities):
    retrieved_entities = set()
    for context in retrieved_contexts:
        if isinstance(context, dict):
            # Extract entities from relevant fields of the context dictionary
            for field in ["title", "plot", "cast"]:
                if field in context:
                    retrieved_entities.update(extract_entities(context[field]))
        else:
            retrieved_entities.update(extract_entities(context))
    recall = len(retrieved_entities.intersection(relevant_entities)) / len(relevant_entities) if relevant_entities else 0
    return recall

def noise_robustness(retrieved_contexts, noisy_queries):
    noise_robustness_score = sum(1 for context in retrieved_contexts if "noise" not in context) / len(noisy_queries)
    return noise_robustness_score

# Generation Metrics using ROUGE
def faithfulness(generated_answers, reference_answers):
    generated_answers, reference_answers = align_lengths(generated_answers, reference_answers)
    logger.info(f"Generated Answers: {generated_answers}")
    logger.info(f"Reference Answers: {reference_answers}")
    scores = rouge.compute(predictions=generated_answers, references=reference_answers)
    logger.info(f"ROUGE Scores for Faithfulness: {scores}")
    faithfulness_score = scores['rouge1']
    return faithfulness_score

def answer_relevance(generated_answers, query):
    scores = rouge.compute(predictions=generated_answers, references=[query] * len(generated_answers))
    logger.info(f"ROUGE Scores for Answer Relevance: {scores}")
    relevance_score = scores['rouge1']
    return relevance_score

def information_integration(generated_answers, reference_answers):
    generated_answers, reference_answers = align_lengths(generated_answers, reference_answers)
    scores = rouge.compute(predictions=generated_answers, references=reference_answers)
    logger.info(f"ROUGE Scores for Information Integration: {scores}")
    integration_score = scores['rouge1']
    return integration_score

def counterfactual_robustness(generated_answers, counterfactual_queries):
    robust_answers = [ans for ans in generated_answers if "counterfactual" not in ans]
    robustness_score = len(robust_answers) / len(generated_answers) if generated_answers else 0
    return robustness_score

def negative_rejection(generated_answers, negative_queries):
    rejected_answers = [ans for ans in generated_answers if "negative" not in ans]
    rejection_score = len(rejected_answers) / len(negative_queries) if negative_queries else 0
    return rejection_score

def measure_latency(func, collection, *args):
    start_time = time.time()
    result = func(*args, collection)
    end_time = time.time()
    return end_time - start_time, result

# Example usage
if __name__ == "__main__":
    queries = ["Tell me about Mad Max: Fury Road.", "Who is the lead in The Terminator?", "What is Kill Bill: Volume 1 about?"]
    
    relevant_contexts = [
        "Mad Max: Fury Road is a 2015 action film directed by George Miller. It features a strong female lead, Imperator Furiosa, played by Charlize Theron.",
        "The Terminator is a 1984 science fiction film directed by James Cameron, featuring Sarah Connor, played by Linda Hamilton, as a strong female lead.",
        "Kill Bill: Volume 1 is a 2003 action film directed by Quentin Tarantino. The film's protagonist, Beatrix Kiddo, played by Uma Thurman, is a strong female lead."
    ]
    
    retrieved_contexts = [
        {"title": "Mad Max: Fury Road", "genre": "Action", "plot": "In a post-apocalyptic wasteland, Max teams up with Furiosa to escape a tyrant who controls the land's water supply.", "cast": "Tom Hardy, Charlize Theron, Nicholas Hoult", "release_date": "2015-05-15", "popularity": 100.0, "vote_average": 7.8},
        {"title": "The Terminator", "genre": "Sci-Fi", "plot": "A cyborg is sent from the future on a deadly mission. He has to kill Sarah Connor, a young woman whose life will have a great significance in years to come.", "cast": "Arnold Schwarzenegger, Linda Hamilton, Michael Biehn", "release_date": "1984-10-26", "popularity": 85.0, "vote_average": 8.0},
        {"title": "Kill Bill: Volume 1", "genre": "Action", "plot": "After awakening from a four-year coma, a former assassin wreaks vengeance on the team of assassins who betrayed her.", "cast": "Uma Thurman, David Carradine, Daryl Hannah", "release_date": "2003-10-10", "popularity": 90.0, "vote_average": 8.1}
    ]

    precision = context_precision(retrieved_contexts, relevant_contexts)
    recall = context_recall(retrieved_contexts, relevant_contexts)
    relevance = context_relevance(retrieved_contexts, queries[0])
    entity_recall = context_entity_recall(retrieved_contexts, {"Charlize Theron", "Linda Hamilton", "Uma Thurman"})
    noise_robustness_score = noise_robustness(retrieved_contexts, ["noisy query 1", "noisy query 2"])

    generated_answers = ["Mad Max: Fury Road features a strong female lead, Furiosa, played by Charlize Theron.", "The Terminator features Sarah Connor, played by Linda Hamilton, as a strong female lead.", "Kill Bill: Volume 1 has Beatrix Kiddo, a strong female lead played by Uma Thurman."]
    reference_answers = ["Mad Max: Fury Road features a strong female lead, Furiosa, played by Charlize Theron.", "The Terminator features Sarah Connor, played by Linda Hamilton, as a strong female lead.", "Kill Bill: Volume 1 has Beatrix Kiddo, a strong female lead played by Uma Thurman."]
    
    counterfactual_queries = ["Who directed The Godfather in 2020?", "What is the plot of Pulp Fiction directed by Steven Spielberg?", "In which year was Inception released by Martin Scorsese?", "Who starred in The Dark Knight directed by Alfred Hitchcock?"]
    negative_queries = ["Tell me about the plot of a non-existent movie.", "What is the release date of the imaginary sequel to Inception?", "Who are the actors in the made-up film 'Random Movie 2025'?", "Describe the storyline of a fictional movie never produced."]

    faithfulness_score = faithfulness(generated_answers, reference_answers)
    answer_relevance_score = answer_relevance(generated_answers, queries[0])
    information_integration_score = information_integration(generated_answers, reference_answers)
    counterfactual_robustness_score = counterfactual_robustness(generated_answers, counterfactual_queries)
    negative_rejection_score = negative_rejection(generated_answers, negative_queries)

    # Assume collection is available here
    from chromadb import Client
    from chromadb.config import Settings
    client = Client(Settings(persist_directory="chroma_db"))
    collection = client.create_collection("movies")

    latencies = [measure_latency(retrieve_context, collection, query) for query in queries]

    print(f"Context Precision: {precision}")
    print(f"Context Recall: {recall}")
    print(f"Context Relevance: {relevance}")
    print(f"Context Entity Recall: {entity_recall}")
    print(f"Noise Robustness: {noise_robustness_score}")
    print(f"Faithfulness: {faithfulness_score}")
    print(f"Answer Relevance: {answer_relevance_score}")
    print(f"Information Integration: {information_integration_score}")
    print(f"Counterfactual Robustness: {counterfactual_robustness_score}")
    print(f"Negative Rejection: {negative_rejection_score}")
    print(f"Average Latency: {np.mean([latency[0] for latency in latencies])} seconds")
1   