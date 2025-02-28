# retrieval.py

import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import logging
from config import *  # Import configurations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# At the top of the file
MODEL_CONFIGS = {
    "blair": {
        "name": "hyp1231/blair-roberta-large",
        "torch_dtype": torch.float16
    },
    "mpnet": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "torch_dtype": None
    }
}

# Module-level variables to store models and their configurations
_models = {}
_tokenizers = {}
def initialize_chromadb(collection_key, database="blair"):
    """
    Initialize ChromaDB client and ensure the corresponding model is loaded.
    """
    # First, ensure the model is loaded
    if database not in _models:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = MODEL_CONFIGS[database]
        
        _tokenizers[database] = AutoTokenizer.from_pretrained(config["name"])
        model_kwargs = {"torch_dtype": config["torch_dtype"]} if config["torch_dtype"] else {}
        _models[database] = AutoModel.from_pretrained(config["name"], **model_kwargs).to(device)
        _models[database].eval()
        
        logger.info(f"Initialized model for database type: {database}")

    # Select the appropriate database configuration
    config = DATASET_CONFIGS_MPNET if database == "mpnet" else DATABASES
    
    if collection_key not in config:
        raise ValueError(f"Collection key '{collection_key}' not found in {database} configuration.")

    db_path = config[collection_key]["db_path"]
    collection_name = config[collection_key]["collection_name"]

    logger.info(f"Initializing ChromaDB client for collection: {collection_key}")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    existing_collections = [col.name for col in chroma_client.list_collections()]
    
    if collection_name in existing_collections:
        collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"Using existing ChromaDB collection: {collection_name}")
    else:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        logger.info(f"Created new ChromaDB collection: {collection_name}")

    return collection

def compute_embedding(text, model_type="blair"):  # Changed default to match your database type
    """
    Compute embedding using the loaded model.
    """
    if model_type not in _models:
        raise ValueError(f"Model {model_type} not initialized. Call initialize_chromadb first.")
    
    model = _models[model_type]
    tokenizer = _tokenizers[model_type]
    device = model.device
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        embedding = embedding / embedding.norm(dim=1, keepdim=True)

    return embedding.cpu().numpy()[0]

def collect_results_per_product(product_names, collection, user_history, max_products=20, max_product_names = 100):
    if not product_names:
        print("No product names provided.")
        return -1  # Return -1 if product_names is empty
    print(f"max product name {max_product_names}")
    product_names = product_names[:max_product_names]
    model_type = "mpnet" if "mpnet" in collection.name.lower() else "blair"
    # Clean product names
    product_names = [name.strip('*').strip() for name in product_names]
    print(f"Cleaned product names: {product_names}")

    doc_distance_map = defaultdict(list)
    final_result = []
    seen_documents = set()
    seen_ids = set(user_history)  # Initialize seen_ids with user_history

    # Step 1: Collect results for each product name
    for product_name in product_names:
        print(f"Processing product name: '{product_name}'")
        try:
            # Compute embedding for the product name
            query_embedding = compute_embedding(product_name, model_type=model_type)
            print(f"Computed embedding for '{product_name}'")

            # Query ChromaDB using the embedding
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=20  # Query 20 items per product name
            )
            print(f"Query results for '{product_name}': {len(results['documents'][0])} items retrieved.")

            # Check if results are empty or documents are None
            if not results['documents'] or not results['documents'][0]:
                print(f"No documents found for '{product_name}'.")
                continue

            # Store documents, distances, and metadatas
            zipped_results = list(
                zip(results['documents'][0], results['distances'][0], results['metadatas'][0])
            )
            doc_distance_map[product_name] = zipped_results
            print(f"Stored {len(zipped_results)} results for '{product_name}'")

        except Exception as e:
            print(f"Error processing product name '{product_name}': {e}")

    # Check if any results were found
    if not doc_distance_map:
        print("No results found for any product names.")
        return -1  # Return -1 if no results were found

    # Step 2: Collect the best item from each product name
    print("Collecting the best item from each product name.")
    for product_name in product_names:
        if product_name in doc_distance_map:
            sorted_results = sorted(doc_distance_map[product_name], key=lambda x: x[1])  # Sort by distance
            for document, distance, metadata in sorted_results:
                product_id = metadata.get("parent_asin")
                if not product_id:
                    print(f"Metadata does not contain a valid ID: {metadata}")
                    continue
                if product_id not in seen_ids:
                    final_result.append((document, distance, product_id))
                    seen_documents.add(document)
                    seen_ids.add(product_id)
                    print(f"Added best document '{document}' for '{product_name}'")
                    break
                else:
                    print(f"Skipping duplicate product ID '{product_id}' for '{product_name}'.")

    # If we have reached max_products, return
    if len(final_result) >= max_products:
        print(f"Reached max products limit of {max_products} after collecting best items.")
        return final_result[:max_products]

    # Step 3: Collect remaining items in a round-robin fashion
    print("Collecting remaining items in a round-robin fashion.")
    index_per_product = {product_name: 0 for product_name in product_names}

    while len(final_result) < max_products:
        added_any = False
        for product_name in product_names:
            if len(final_result) >= max_products:
                break
            if product_name in doc_distance_map:
                sorted_results = sorted(doc_distance_map[product_name], key=lambda x: x[1])
                idx = index_per_product[product_name]
                while idx < len(sorted_results):
                    document, distance, metadata = sorted_results[idx]
                    idx += 1
                    product_id = metadata.get("parent_asin")
                    if not product_id:
                        print(f"Metadata does not contain a valid ID: {metadata}")
                        continue
                    if product_id not in seen_ids:
                        final_result.append((document, distance, product_id))
                        seen_documents.add(document)
                        seen_ids.add(product_id)
                        index_per_product[product_name] = idx
                        added_any = True
                        print(f"Added document '{document}' for '{product_name}'")
                        break
                    else:
                        print(f"Skipping duplicate product ID '{product_id}' for '{product_name}'.")
                index_per_product[product_name] = idx
        if not added_any:
            print("No more unique items to add. Ending collection.")
            break

    print(f"Final result contains {len(final_result)} items.")
    return final_result[:max_products]


def collect_results_per_product_top4(product_names, collection, user_history, max_products=20):
    if not product_names:
        print("No product names provided.")
        return -1  # Return -1 if product_names is empty
    model_type = "mpnet" if "mpnet" in collection.name.lower() else "blair"
    # Clean product names
    product_names = [name.strip('*').strip() for name in product_names]
    print(f"Cleaned product names: {product_names}")

    doc_distance_map = defaultdict(list)
    final_result = []
    seen_documents = set()
    seen_ids = set(user_history)  # Initialize seen_ids with user_history

    # Step 1: Collect results for each product name
    for product_name in product_names:
        print(f"Processing product name: '{product_name}'")
        try:
            # Compute embedding for the product name
            query_embedding = compute_embedding(product_name, model_type=model_type)
            print(f"Computed embedding for '{product_name}'")

            # Query ChromaDB using the embedding
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=20  # Query 20 items per product name
            )
            print(f"Query results for '{product_name}': {len(results['documents'][0])} items retrieved.")

            # Check if results are empty or documents are None
            if not results['documents'] or not results['documents'][0]:
                print(f"No documents found for '{product_name}'.")
                continue

            # Store documents, distances, and metadatas
            zipped_results = list(
                zip(results['documents'][0], results['distances'][0], results['metadatas'][0])
            )
            doc_distance_map[product_name] = zipped_results
            print(f"Stored {len(zipped_results)} results for '{product_name}'")

        except Exception as e:
            print(f"Error processing product name '{product_name}': {e}")

    # Check if any results were found
    if not doc_distance_map:
        print("No results found for any product names.")
        return -1  # Return -1 if no results were found

    # Step 2: Collect top 4 items from each of the first 5 products
    print("Collecting top 4 items from each of the first 5 products.")
    products_processed = 0
    for product_name in product_names[:5]:  # Only process first 5 products
        if product_name in doc_distance_map:
            items_added = 0
            sorted_results = sorted(doc_distance_map[product_name], key=lambda x: x[1])  # Sort by distance
            for document, distance, metadata in sorted_results:
                if items_added >= 4:  # Only take top 4 items per product
                    break
                if len(final_result) >= max_products:
                    break
                product_id = metadata.get("parent_asin")
                if not product_id:
                    print(f"Metadata does not contain a valid ID: {metadata}")
                    continue
                if product_id not in seen_ids:
                    final_result.append((document, distance, product_id))
                    seen_documents.add(document)
                    seen_ids.add(product_id)
                    items_added += 1
                    print(f"Added document '{document}' for '{product_name}' (item {items_added}/4)")
                else:
                    print(f"Skipping duplicate product ID '{product_id}' for '{product_name}'.")
            products_processed += 1

    # If we have reached max_products, return
    if len(final_result) >= max_products:
        print(f"Reached max products limit of {max_products} after collecting top 4 items.")
        return final_result[:max_products]

    # Step 3: If we still need more items, collect remaining items in a round-robin fashion
    if len(final_result) < max_products:
        print("Collecting remaining items in a round-robin fashion.")
        index_per_product = {product_name: 0 for product_name in product_names}

        while len(final_result) < max_products:
            added_any = False
            for product_name in product_names:
                if len(final_result) >= max_products:
                    break
                if product_name in doc_distance_map:
                    sorted_results = sorted(doc_distance_map[product_name], key=lambda x: x[1])
                    idx = index_per_product[product_name]
                    while idx < len(sorted_results):
                        document, distance, metadata = sorted_results[idx]
                        idx += 1
                        product_id = metadata.get("parent_asin")
                        if not product_id:
                            print(f"Metadata does not contain a valid ID: {metadata}")
                            continue
                        if product_id not in seen_ids:
                            final_result.append((document, distance, product_id))
                            seen_documents.add(document)
                            seen_ids.add(product_id)
                            index_per_product[product_name] = idx
                            added_any = True
                            print(f"Added document '{document}' for '{product_name}'")
                            break
                        else:
                            print(f"Skipping duplicate product ID '{product_id}' for '{product_name}'.")
                    index_per_product[product_name] = idx
            if not added_any:
                print("No more unique items to add. Ending collection.")
                break

    print(f"Final result contains {len(final_result)} items.")
    return final_result[:max_products]