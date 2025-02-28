#!/usr/bin/env python3
import os
import sys
import argparse
import gzip
import shutil
import json
import pandas as pd
from collections import defaultdict, Counter

########################################
# Utility Functions
########################################

def extract_gz_to_file(gz_file_path, output_file_path=None):
    """
    Extracts a .gz (gzip) file and saves it as a new file.
    """
    if not os.path.isfile(gz_file_path):
        raise FileNotFoundError(f"The file {gz_file_path} does not exist.")

    if output_file_path is None:
        output_file_path = gz_file_path.rstrip('.gz')

    try:
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(output_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extraction complete: {output_file_path}")
        return output_file_path
    except OSError as e:
        raise OSError(f"Error during extraction: {e}")

def remove_duplicates(df, subset_cols, keep='first'):
    before = len(df)
    df_cleaned = df.drop_duplicates(subset=subset_cols, keep=keep).reset_index(drop=True)
    after = len(df_cleaned)
    print(f"Duplicates removed: {before - after}")
    return df_cleaned

def assign_split(df):
    identifiers = set(zip(df['user_id'], df['parent_asin'], df['timestamp']))
    return identifiers

def determine_split(row, train_ids, test_ids, val_ids):
    identifier = (row['user_id'], row['parent_asin'], row['timestamp'])
    if identifier in train_ids:
        return 'train'
    elif identifier in test_ids:
        return 'test'
    elif identifier in val_ids:
        return 'val'
    else:
        return 'unknown'

def create_json_split(df, split_label):
    split_df = df[df['split'] == split_label]
    user_reviews = defaultdict(list)
    
    for _, row in split_df.iterrows():
        review = {
            "product_name": row['product_name'],
            "parent_asin": row['parent_asin'],
            "rating": row['rating'],
            "title": row['title'],
            "text": row['text'],
            "timestamp": row['timestamp']
        }
        user_reviews[row['user_id']].append(review)
    
    output = [{"user_id": user_id, "reviews": reviews} for user_id, reviews in user_reviews.items()]
    return output

def merge_json(data1, data2):
    user_map = {user['user_id']: user for user in data1}
    for user in data2:
        user_id = user['user_id']
        if user_id in user_map:
            user_map[user_id]['reviews'].extend(user['reviews'])
        else:
            data1.append(user)
    return data1

def sort_reviews_by_timestamp(data):
    for user in data:
        user['reviews'].sort(key=lambda review: review['timestamp'])
    return data

def combine_train_val(train_json, val_json):
    combined_data = merge_json(train_json, val_json)
    combined_data = sort_reviews_by_timestamp(combined_data)
    return combined_data

def filter_reviews_more_than_3_interactions(input_file, output_file):
    reviews = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
                if 'user_id' not in review or 'parent_asin' not in review:
                    continue
                reviews.append(review)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue

    print(f"Total reviews loaded for filtering: {len(reviews)}")

    user_counts = Counter(r['user_id'] for r in reviews)
    item_counts = Counter(r['parent_asin'] for r in reviews)

    filtered_reviews = [
        r for r in reviews
        if user_counts[r['user_id']] > 3 and item_counts[r['parent_asin']] > 3
    ]

    print(f"Filtered reviews: {len(filtered_reviews)} out of {len(reviews)}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for r in filtered_reviews:
            json.dump(r, out_f, ensure_ascii=False)
            out_f.write('\n')

    print(f"Filtered reviews saved to {output_file}")
    return output_file

def load_meta(meta_file):
    meta_map = {}
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                parent_asin = record.get('parent_asin')
                title = record.get('title')
                if parent_asin and title:
                    meta_map[parent_asin] = title
            except json.JSONDecodeError as e:
                print(f"Error decoding meta JSON: {e}")
                continue
    print(f"Loaded {len(meta_map)} meta entries.")
    return meta_map

def add_product_names_from_meta(df, meta_map):
    df['product_name'] = df.apply(
        lambda row: meta_map[row['parent_asin']] if row['parent_asin'] in meta_map else row.get('product_name', row['asin']),
        axis=1
    )
    return df

def main(args):
    os.makedirs('data', exist_ok=True)

    dataset_name = args.dataset_name
    no_meta = args.no_meta
    combine = args.combine_train_val

    # Paths
    reviews_gz = f"data/{dataset_name}.jsonl.gz"
    meta_gz = f"data/meta_{dataset_name}.jsonl.gz"
    train_gz = f"data/{dataset_name}.train.csv.gz"
    test_gz = f"data/{dataset_name}.test.csv.gz"
    val_gz = f"data/{dataset_name}.valid.csv.gz"

    reviews_file = f"data/{dataset_name}.jsonl"
    meta_file = f"data/meta_{dataset_name}.jsonl"
    filtered_output = f"data/{dataset_name}_more_than_3_with_product.jsonl"

    train_file = f"data/{dataset_name}.train.csv"
    test_file = f"data/{dataset_name}.test.csv"
    val_file = f"data/{dataset_name}.valid.csv"

    train_output = "data/train_output.json"
    test_output = "data/test_output.json"
    val_output = "data/val_output.json"
    combined_train_val_output = "data/combined_train_val_output.json"

    # Step 1: Extract all gz files
    if not os.path.isfile(reviews_file):
        extract_gz_to_file(reviews_gz, reviews_file)

    # Meta extraction if not no_meta and meta_gz exists
    meta_available = False
    if not no_meta and os.path.isfile(meta_gz):
        extract_gz_to_file(meta_gz, meta_file)
        meta_available = True
    else:
        print("No meta file or --no_meta specified. Skipping meta product name enrichment.")

    # Extract splits
    if not os.path.isfile(train_file):
        extract_gz_to_file(train_gz, train_file)
    if not os.path.isfile(test_file):
        extract_gz_to_file(test_gz, test_file)
    if not os.path.isfile(val_file):
        extract_gz_to_file(val_gz, val_file)

    # Step 2: Filter reviews >3 interactions
    filtered_reviews_file = filter_reviews_more_than_3_interactions(reviews_file, filtered_output)

    # Step 3: Load filtered reviews
    filtered_reviews = []
    with open(filtered_reviews_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            review = json.loads(line)
            filtered_reviews.append(review)
    reviews_df = pd.DataFrame(filtered_reviews)
    print(f"Total filtered reviews loaded into DataFrame: {len(reviews_df)}")

    if 'product_name' not in reviews_df.columns:
        print("'product_name' is missing in filtered reviews. Using 'asin' as 'product_name'.")
        reviews_df['product_name'] = reviews_df['asin']
    else:
        print("'product_name' is present in filtered reviews.")

    # If meta available, load and add product names
    if meta_available and os.path.isfile(meta_file):
        meta_map = load_meta(meta_file)
        reviews_df = add_product_names_from_meta(reviews_df, meta_map)

    # Step 4: Load splits
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    val_df = pd.read_csv(val_file)

    print(f"Train reviews: {len(train_df)}")
    print(f"Test reviews: {len(test_df)}")
    print(f"Validation reviews: {len(val_df)}")

    # Step 5: Remove duplicates
    duplicate_subset = ['user_id', 'parent_asin', 'timestamp']
    reviews_df = remove_duplicates(reviews_df, subset_cols=duplicate_subset, keep='first')
    train_df = remove_duplicates(train_df, subset_cols=duplicate_subset, keep='first')
    test_df = remove_duplicates(test_df, subset_cols=duplicate_subset, keep='first')
    val_df = remove_duplicates(val_df, subset_cols=duplicate_subset, keep='first')

    # Step 6: Assign splits
    reviews_df['split'] = None
    train_ids = assign_split(train_df)
    test_ids = assign_split(test_df)
    val_ids = assign_split(val_df)

    reviews_df['split'] = reviews_df.apply(lambda row: determine_split(row, train_ids, test_ids, val_ids), axis=1)
    print(reviews_df['split'].value_counts())

    # Step 7: Create JSON outputs for each split
    train_json = create_json_split(reviews_df, 'train')
    test_json = create_json_split(reviews_df, 'test')
    val_json = create_json_split(reviews_df, 'val')

    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=False, indent=4)

    with open(test_output, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False, indent=4)

    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_json, f, ensure_ascii=False, indent=4)

    print("Train/Test/Val JSON files saved successfully.")

    # Combine train and val if requested
    if combine:
        combined_train_val_data = combine_train_val(train_json, val_json)
        with open(combined_train_val_output, 'w', encoding='utf-8') as f:
            json.dump(combined_train_val_data, f, ensure_ascii=False, indent=4)
        print(f"Combined train and val data saved to {combined_train_val_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete data preprocessing pipeline. All input/output in `data` folder, minimal arguments.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., Video_Games)')
    parser.add_argument('--no_meta', action='store_true', help='If set, skip using meta file for product names')
    parser.add_argument('--combine_train_val', action='store_true', help='If set, combine train and val sets into one file')

    args = parser.parse_args()
    main(args)
