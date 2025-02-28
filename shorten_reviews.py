import json
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def process_reviews(input_file, output_file):
    """
    Shorten reviews in a JSON file using TF-IDF and save to a new JSON file.
    
    :param input_file: Path to the input JSON file containing reviews.
    :param output_file: Path to the output JSON file for saving shortened reviews.
    """
    # Load data from JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten data to create a DataFrame
    review_rows = []
    for user in data:
        user_id = user['user_id']
        for review in user['reviews']:
            review_rows.append({
                'user_id': user_id,
                'text': review['text'],  # Review content
                'parent_asin': review['parent_asin']  # Product ID
            })

    # Create DataFrame
    reviews_df = pd.DataFrame(review_rows)

    # Initialize TfidfVectorizer to calculate IDF scores
    vectorizer = TfidfVectorizer(use_idf=True, stop_words='english', max_df=0.95, min_df=2)
    vectorizer.fit(reviews_df['text'])  # Fit on all review texts to get IDF values

    # Extract vocabulary and IDF scores
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    # Function to calculate sentence weight based on TF-IDF and POS tagging
    def calculate_sentence_weight(sentence, idf_scores):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        descriptive_weight = sum(
            idf_scores.get(word.lower(), 0)
            for word, tag in tagged_words
            if tag.startswith('NN') or tag.startswith('JJ')  # Prioritize nouns and adjectives
        )
        return descriptive_weight / max(len(words), 1)

    # Process each review to shorten it
    for index, row in reviews_df.iterrows():
        sentences = sent_tokenize(row['text'])  # Tokenize text into sentences
        weighted_sentences = [
            (sentence, calculate_sentence_weight(sentence, idf_scores)) for sentence in sentences
        ]

        # Sort sentences by weight
        sorted_sentences = sorted(weighted_sentences, key=lambda x: x[1], reverse=True)
        
        # Select top sentences within a token limit
        selected_sentences = []
        token_count = 0
        for sentence, weight in sorted_sentences:
            if sentence in selected_sentences:
                continue
            sentence_tokens = word_tokenize(sentence)
            token_count += len(sentence_tokens)
            if token_count > 128:  # Limit to 128 tokens
                break
            selected_sentences.append(sentence)
        
        # Fallback if no sentences were selected
        if not selected_sentences:
            selected_sentences.append(sorted_sentences[0][0] if sorted_sentences else "No relevant information available.")

        # Maintain original order of selected sentences
        selected_sentences = sorted(selected_sentences, key=lambda s: sentences.index(s))
        reviews_df.at[index, 'selected_sentences'] = " ".join(selected_sentences)

    # Create a lookup for selected sentences by user_id and parent_asin
    selected_sentences_lookup = {
        (row['user_id'], row['parent_asin']): row['selected_sentences']
        for _, row in reviews_df.iterrows()
    }

    # Function to update reviews with shortened text
    def update_reviews(data):
        for user_data in data:
            user_id = user_data['user_id']
            for review in user_data['reviews']:
                parent_asin = review['parent_asin']
                new_text = selected_sentences_lookup.get((user_id, parent_asin))
                if new_text:
                    review['text'] = new_text  # Replace old text with shortened version

    # Update reviews and save to JSON file
    update_reviews(data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Shortened reviews saved to {output_file}.")

def validate_review_counts(input_file, output_file):
    """
    Validate that the number of reviews in the input and output files match.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSON file.
    :return: Boolean indicating whether the counts match.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    input_count = sum(len(user['reviews']) for user in input_data)
    output_count = sum(len(user['reviews']) for user in output_data)

    if input_count == output_count:
        print(f"Validation passed: {input_count} reviews in both input and output files.")
        return True
    else:
        print(f"Validation failed: {input_count} reviews in input file, {output_count} reviews in output file.")
        return False

# Example usage
if __name__ == "__main__":
    input_file = 'new_data/Video_Games.reduced_300_users.json'
    output_file = 'new_data/Video_Games.shortened_reduced_300_users.json'
    
    # Process reviews and save to a new file
    process_reviews(input_file, output_file)
    
    # Validate review counts
    validate_review_counts(input_file, output_file)
