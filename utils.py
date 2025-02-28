# utils.py
import re

def extract_latest_n_reviews(data, n):
    """Extract the latest 'n' reviews from the data."""
    review = []
    for user in data:
        reviews = user['reviews']
        # Ensure reviews are sorted by timestamp (earliest to latest)
        sorted_reviews = sorted(reviews, key=lambda x: x['timestamp'])
        # Get the last 'n' reviews
        latest_reviews = sorted_reviews[-n:]
        review.extend(latest_reviews)
    return review

import random

def extract_latest_n_reviews(data, n, only_positive=False, shuffle_reviews=False):
    """
    Extract the latest 'n' reviews from each user in 'data'. 
    Optionally, only return reviews with a rating above 4.0, 
    and/or shuffle the final list of reviews.
    
    :param data: A list of user dictionaries, 
                 each containing a 'reviews' key with a list of reviews.
    :param n: The number of latest reviews to extract per user.
    :param only_positive: If True, only reviews with rating > 4.0 are returned.
    :param shuffle_reviews: If True, shuffle the final list of extracted reviews.
    :return: A list of reviews.
    """
    all_reviews = []
    
    for user in data:
        # Get all reviews for this user
        reviews = user['reviews']
        
        # If only_positive is True, filter out reviews where rating <= 4.0
        if only_positive:
            reviews = [r for r in reviews if r.get('rating', 0) >= 4.0]

        # Ensure reviews are sorted by timestamp (earliest to latest)
        sorted_reviews = sorted(reviews, key=lambda x: x['timestamp'])
        
        # Get the last 'n' reviews
        latest_reviews = sorted_reviews[-n:]
        all_reviews.extend(latest_reviews)
    
    # Shuffle reviews if shuffle_reviews is True
    if shuffle_reviews:
        random.shuffle(all_reviews)
        
    return all_reviews


import re

def extract_product_names_from_combined(response_text):
    """
    Extract product names from the response text after detecting keywords containing
    'candidate' or 'category'. Handles various list formats including items in individual
    square brackets separated by commas.
    """
    product_names = []

    # Split the text into lines
    lines = response_text.strip().split('\n')
    start_extracting = False

    for line in lines:
        stripped_line = line.strip()

        if not start_extracting:
            # Check if the line contains any form of 'candidate' or 'category' (case-insensitive)
            keyword_match = re.search(r'(candidate|category|categories)', stripped_line, re.IGNORECASE)
            if keyword_match:
                start_extracting = True
                # Extract items from the line if any, excluding 'username'
                items_in_brackets = re.findall(r'\[([^\]]+)\]', stripped_line)
                if items_in_brackets:
                    for item in items_in_brackets:
                        item = item.strip()
                        if item.lower() != 'username':
                            product_names.append(item)
                continue  # Proceed to next line
            continue  # Skip to the next line until the keyword is found

        # Skip empty lines
        if not stripped_line:
            continue

        # Attempt to extract product names

        # Handle numbered list items with bolded product names
        match_numbered_bold_item = re.match(r'^\d+[\.\)-]?\s*\*\*(.+?)\*\*', stripped_line)
        if match_numbered_bold_item:
            item_text = match_numbered_bold_item.group(1).strip()
            product_names.append(item_text)
            continue

        # Handle lines with items in square brackets, excluding 'username'
        items_in_brackets = re.findall(r'\[([^\]]+)\]', stripped_line)
        if items_in_brackets:
            for item in items_in_brackets:
                item = item.strip()
                if item.lower() != 'username':
                    product_names.append(item)
            continue

        # Handle lines with semicolon-separated items
        if ';' in stripped_line:
            items = [item.strip() for item in stripped_line.split(';') if item.strip()]
            product_names.extend(items)
            continue

        # Handle numbered list items with optional dot and space
        match_numbered_item = re.match(r'^\d+[\.\)-]?\s*"?(.+?)"?$', stripped_line)
        if match_numbered_item:
            item_text = match_numbered_item.group(1).strip('"').strip()
            if item_text.lower() != 'username':
                product_names.append(item_text)
            continue

        # Handle bulleted list items starting with "-", "*", "+"
        match_bullet_item = re.match(r'^[-*+]\s+(.*)', stripped_line)
        if match_bullet_item:
            item_text = match_bullet_item.group(1).strip()
            product_names.append(item_text)
            continue

        # Handle standalone quoted items
        match_quoted_line = re.match(r'^"(.+?)"$', stripped_line)
        if match_quoted_line:
            item_text = match_quoted_line.group(1).strip()
            product_names.append(item_text)
            continue

        # Stop extraction if a new section is detected
        if re.match(r'^[A-Z][A-Za-z0-9_\s]*:$', stripped_line):
            break

    return product_names





import re

import re

def extract_product_names_adapter(response_text):
    """Extract product names from the response text based on list indicators and remove trailing '**' if present."""
    product_names = []
    lines = response_text.strip().split('\n')
    start_extracting = False

    # Pattern to detect list items (numbered or bullet points)
    list_item_pattern = re.compile(r'^(\d+[\.\)-]?|[-*+])\s+(.*)')

    for line in lines:
        stripped_line = line.strip()

        if not start_extracting:
            # Check if the line starts with a list indicator
            if list_item_pattern.match(stripped_line):
                start_extracting = True
            else:
                continue  # Skip lines until we find the start of the list

        # Now we are in extraction mode
        match_list_item = list_item_pattern.match(stripped_line)
        if match_list_item:
            # Extract the text after the list indicator
            item_text = match_list_item.group(2).strip()
            product_names.append(item_text)
        else:
            # Handle continuation lines (non-empty lines after a list item)
            if stripped_line == '':
                continue  # Skip empty lines
            else:
                # Append to the previous item if it's a continuation
                if product_names:
                    product_names[-1] += ' ' + stripped_line
                else:
                    continue  # Skip if there's no previous item

    # Helper function to remove leading/trailing asterisks
    def remove_leading_trailing_asterisks(text):
        # Remove asterisks from the start
        while text.startswith('*'):
            text = text[1:]
        # Remove asterisks from the end
        while text.endswith('*'):
            text = text[:-1]
        return text.strip()

    # Final pass: remove extra '*' characters from each product name
    for i in range(len(product_names)):
        product_names[i] = remove_leading_trailing_asterisks(product_names[i])

    return product_names




import re
def extract_product_names_alpaca(response_text):
    """
    Extract product names by looking for 'Candidate Item Categories:' first.
    If found, start extracting from the lines after it using adapter-style list parsing.
    If not found, fall back to the original adapter logic from the start of the text.
    """
    lines = response_text.strip().split('\n')
    product_names = []

    # Regex for numbered or bullet-list items:
    # Matches patterns like:
    #   1. Item, 1) Item, 1- Item
    #   - Item, * Item, + Item
    list_item_pattern = re.compile(r'^(\d+[\.\)-]?|[-*+])\s+(.*)')

    # Helper function to remove leading/trailing asterisks
    def remove_leading_trailing_asterisks(text):
        while text.startswith('*'):
            text = text[1:]
        while text.endswith('*'):
            text = text[:-1]
        return text.strip()

    # ------------------------------------------
    # 1) Check if "Candidate Item Categories:" exists in the text
    #    If found, note its line index
    # ------------------------------------------
    candidate_idx = None
    for i, line in enumerate(lines):
        if "Candidate Item Categories:" in line:
            candidate_idx = i
            break

    # ------------------------------------------
    # 2) Define a nested function that does the "adapter" extraction logic
    #    on a given list of lines.
    # ------------------------------------------
    def extract_with_adapter_logic(lines_subset):
        """
        Applies the same logic as 'extract_product_names_adapter' to the provided lines.
        Returns a list of extracted items.
        """
        extracted = []
        start_extracting = False

        for line in lines_subset:
            stripped_line = line.strip()
            if not start_extracting:
                # Check if the line starts with a list indicator
                if list_item_pattern.match(stripped_line):
                    start_extracting = True
                else:
                    continue  # Skip until we find the start of the list

            # We are in extraction mode now
            match_list_item = list_item_pattern.match(stripped_line)
            if match_list_item:
                # Extract the text after the list indicator
                item_text = match_list_item.group(2).strip()
                extracted.append(item_text)
            else:
                # Handle continuation lines (non-empty lines after a list item)
                if stripped_line == '':
                    continue
                else:
                    if extracted:
                        extracted[-1] += ' ' + stripped_line
                    else:
                        continue
        return extracted

    # ------------------------------------------
    # 3) If "Candidate Item Categories:" found, apply adapter logic
    #    starting from the line after candidate_idx
    #    Otherwise, apply adapter logic to the entire text.
    # ------------------------------------------
    if candidate_idx is not None:
        # Start from the line after "Candidate Item Categories:"
        subset_lines = lines[candidate_idx + 1 :]
        product_names = extract_with_adapter_logic(subset_lines)
    else:
        # Fallback: adapter logic on the entire text
        product_names = extract_with_adapter_logic(lines)

    # ------------------------------------------
    # 4) Clean up asterisks
    # ------------------------------------------
    for i in range(len(product_names)):
        product_names[i] = remove_leading_trailing_asterisks(product_names[i])

    return product_names






# utils.py

import re
import numpy as np
from difflib import SequenceMatcher

def normalize(text):
    """
    Normalizes the input text by converting to lowercase, removing punctuation, and extra whitespace.
    
    Args:
        text (str): The input text to normalize.
    
    Returns:
        str: The normalized text.
    """
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_similarity(str1, str2):
    """
    Computes the similarity between two strings as a percentage using difflib's SequenceMatcher.
    
    Args:
        str1 (str): The first string.
        str2 (str): The second string.
    
    Returns:
        float: The similarity percentage between 0 and 100.
    """
    ratio = SequenceMatcher(None, str1, str2).ratio()
    return ratio * 100  # Convert to percentage

def recall_at_k(matches, k):
    """
    Computes Recall@K given the list of matches and K.
    
    Args:
        matches (list of bool): List indicating whether each recommendation is a match.
        k (int): The value of K for Recall@K.
    
    Returns:
        float: The Recall@K value.
    """
    relevant_items = 1  # Assuming there is one relevant item (the test product)
    retrieved_relevant = sum(matches[:k])
    recall = retrieved_relevant / relevant_items
    return min(recall, 1.0)  # Recall cannot exceed 1.0

def dcg_at_k(scores, k):
    """
    Computes DCG@K for the given list of scores.
    
    Args:
        scores (list of float): Relevance scores (e.g., matches as 1.0 or 0.0).
        k (int): The value of K for DCG@K.
    
    Returns:
        float: The DCG@K value.
    """
    dcg = 0.0
    for i, score in enumerate(scores[:k]):
        if score != 0:
            dcg += score / np.log2(i + 2)  # positions are 1-based
    return dcg

def ndcg_at_k(matches, k):
    """
    Computes NDCG@K given the list of matches and K.
    
    Args:
        matches (list of bool): List indicating whether each recommendation is a match.
        k (int): The value of K for NDCG@K.
    
    Returns:
        float: The NDCG@K value.
    """
    # Relevance scores: 1 for match, 0 for non-match
    relevance_scores = [1.0 if match else 0.0 for match in matches]
    dcg = dcg_at_k(relevance_scores, k)
    # Ideal DCG (IDCG) is when the relevant items are ranked at the top
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    if idcg == 0:
        return 0.0
    else:
        return dcg / idcg
