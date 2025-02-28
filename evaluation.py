# evaluation.py
import re
import string
from Levenshtein import ratio
import numpy as np

def normalize(text):
    """Normalize text by converting to lowercase and removing punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('\s+', ' ', text).strip()
    return text

def compute_similarity(str1, str2):
    """Compute similarity between two strings using Levenshtein ratio."""
    return ratio(str1, str2) * 100  # Convert to percentage

def recall_at_k(matches, k):
    """Calculate Recall@K."""
    relevant_in_top_k = matches[:k].count(True)
    total_relevant = 1  # Assuming one relevant item per user
    recall = relevant_in_top_k / total_relevant
    return recall

def ndcg_at_k(matches, k):
    """Calculate NDCG@K."""
    relevance_scores = [1 if match else 0 for match in matches[:k]]
    relevance_scores = np.array(relevance_scores)

    gains = 2 ** relevance_scores - 1
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    dcg = np.sum(gains / discounts)

    ideal_relevance = np.sort(relevance_scores)[::-1]
    ideal_gains = 2 ** ideal_relevance - 1
    idcg = np.sum(ideal_gains / discounts)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg
