import time
import numpy as np
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from evaluate import load

# Load the ROUGE metric from Hugging Face
rouge = load("rouge")

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample functions to retrieve context and generate answers
def retrieve_context(query, collection, n_results=3):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    contexts = [metadata for result in results['metadatas'] for metadata in result]
    return contexts

def generate_answer(contexts, query):
    answer = "Generated answer based on contexts and query"
    return answer

# Placeholder function to extract entities from a given context
def extract_entities(context):
    return ["entity1", "entity2"]

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
    scores = rouge.compute(predictions=formatted_retrieved, references=relevant_contexts)
    precision = scores['precision'] if 'precision' in scores else 0
    return precision

# Custom function to calculate context recall using ROUGE
def context_recall(retrieved_contexts, relevant_contexts):
    formatted_retrieved = format_recommendations(retrieved_contexts)
    formatted_retrieved, relevant_contexts = align_lengths(formatted_retrieved, relevant_contexts)
    scores = rouge.compute(predictions=formatted_retrieved, references=relevant_contexts)
    recall = scores['recall'] if 'recall' in scores else 0
    return recall

# Custom function to calculate context relevance using ROUGE
def context_relevance(retrieved_contexts, query):
    formatted_retrieved = format_recommendations(retrieved_contexts)
    scores = rouge.compute(predictions=formatted_retrieved, references=[query] * len(formatted_retrieved))
    relevance = scores['fmeasure'] if 'fmeasure' in scores else 0
    return relevance

def context_entity_recall(retrieved_contexts, relevant_entities):
    retrieved_entities = set(entity for context in retrieved_contexts for entity in extract_entities(context))
    recall = len(retrieved_entities.intersection(relevant_entities)) / len(relevant_entities) if relevant_entities else 0
    return recall

def noise_robustness(retrieved_contexts, noisy_queries):
    noise_robustness_score = sum(1 for context in retrieved_contexts if "noise" not in context) / len(noisy_queries)
    return noise_robustness_score

# Generation Metrics using ROUGE
def faithfulness(generated_answers, reference_answers):
    generated_answers, reference_answers = align_lengths(generated_answers, reference_answers)
    scores = rouge.compute(predictions=generated_answers, references=reference_answers)
    faithfulness_score = scores['precision'] if 'precision' in scores else 0
    return faithfulness_score

def answer_relevance(generated_answers, query):
    scores = rouge.compute(predictions=generated_answers, references=[query] * len(generated_answers))
    relevance_score = scores['fmeasure'] if 'fmeasure' in scores else 0
    return relevance_score

def information_integration(generated_answers, reference_answers):
    generated_answers, reference_answers = align_lengths(generated_answers, reference_answers)
    scores = rouge.compute(predictions=generated_answers, references=reference_answers)
    integration_score = scores['fmeasure'] if 'fmeasure' in scores else 0
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
    queries = ["what is the capital of india?", ""]  # Your test queries
    relevant_contexts = ["Sample context 1 for query", "Sample context 2 for query"]  # Your ground truth
    retrieved_contexts = ["Sample context 1 for query", "Irrelevant context"]  # Retrieved contexts by your system

    precision = context_precision(retrieved_contexts, relevant_contexts)
    recall = context_recall(retrieved_contexts, relevant_contexts)
    relevance = context_relevance(retrieved_contexts, queries[0])
    entity_recall = context_entity_recall(retrieved_contexts, {"entity1", "entity2"})
    noise_robustness_score = noise_robustness(retrieved_contexts, ["noisy query 1", "noisy query 2"])

    generated_answers = ["Generated answer 1", "Generated answer 2"]  # Dummy generated answers
    reference_answers = ["Generated answer 1", "Generated answer 2"]
    counterfactual_queries = ["Counterfactual query 1", "Counterfactual query 2"]
    negative_queries = ["Negative query 1", "Negative query 2"]

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
