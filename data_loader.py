# data_loader.py
import json

def load_user_reviews(file_path):
    """Load user reviews from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
