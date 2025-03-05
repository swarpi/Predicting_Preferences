import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")

# config.py

# Language model parameters
TOKENIZER_PATH = "models/hf-frompretrained-download/meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH = "models/hf-frompretrained-downloadmeta-llama/Meta-Llama-3-8B-Instruct"

def get_model_path_adapter_user_profile(sample_size):
    return f"outputs/adapter_test_user_profile_epoch_{QLORA_PARAMS['lora_num_epochs']}_{sample_size}_chatgpt_data_samples"

def get_model_path_adapter_candidate_items(sample_size):
    return f"outputs/adapter_test_candidate_items_epoch_{QLORA_PARAMS['lora_num_epochs']}_{sample_size}_chatgpt_data_samples"

# Dynamic model path for experiments with different sample sizes
def get_model_path_user_profile(sample_size):
    return f"outputs/best_model_{sample_size}_samples"

def get_tokenizer_path_user_profile(sample_size):
    return f"outputs/best_model_{sample_size}_samples"

# Dynamic model path for experiments with different sample sizes
def get_model_path_user_profile_and_candidate_items(sample_size):
    return f"outputs/best_model_up_ci_{sample_size}_samples"

def get_tokenizer_path_user_profile_and_candidate_items(sample_size):
    return f"outputs/best_model_up_ci_{sample_size}_samples"

def get_adapter_name_user_profile(sample_size, dataset, data_source):
    """
    Returns the directory name for the user_profile adapter 
    based on sample_size, dataset, and data_source.
    Example result: 
        "outputs/adapter_test_user_profile_epoch_2_100_chatgpt_data_samples_video_games"
    """
    base = f"outputs/main/adapter_test_user_profile_epoch_{QLORA_PARAMS['lora_num_epochs']}_{sample_size}_{data_source}"
    if dataset == "video_games":
        base += "_video_games"
    # If dataset == "beauty", we leave it as is.
    return base


def get_adapter_name_candidate_items(sample_size, dataset, data_source):
    """
    Returns the directory name for the candidate_items adapter 
    based on sample_size, dataset, and data_source.
    Example result:
        "outputs/adapter_test_candidate_items_epoch_2_100_pipeline_data_samples_video_games"
    """
    base = f"outputs/main/adapter_test_candidate_items_epoch_{QLORA_PARAMS['lora_num_epochs']}_{sample_size}_{data_source}"
    if dataset == "video_games":
        base += "_video_games"
    return base

DATABASES = {
    "video_games": {
        "db_path": "./chroma_db_video_games",
        "collection_name": "video_games_product_embeddings_filtered",
        "data_paths": {
            "reduced_file": os.path.join(DATA_DIR, "video_games_combined.json"),
            "meta_file": os.path.join(RAW_DATA_DIR, "meta_Video_Games.jsonl")
        }
    },
    "beauty": {
        "db_path": "./chroma_db_beauty",
        "collection_name": "beauty_product_embeddings_filtered",
        "data_paths": {
            "reduced_file": os.path.join(DATA_DIR, "Beauty_combined.json"),
            "meta_file": os.path.join(RAW_DATA_DIR, "meta_All_Beauty.jsonl")
        }
    }
}

DATASET_CONFIGS_MPNET = {
    "video_games": {
        "db_path": "./chroma_db_video_games_mpnet",
        "collection_name": "video_games_product_embeddings_mpnet"
    },
    "beauty": {
        "db_path": "./chroma_db_beauty_mpnet",
        "collection_name": "beauty_product_embeddings_mpnet"
    }
}

PIPELINE_PARAMS = {
    'max_length': 4096,
    'num_return_sequences': 1,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.95,
    'repetition_penalty': 1.2,
            # 'pad_token_id' will be set in the code after tokenizer is initialized
}
# QLoRA Fine-tuning Parameters
QLORA_PARAMS = {
    'lora_r': 8,
    'lora_alpha': 8,
    'lora_dropout': 0.01,
    'lora_target_modules': ['q_proj', 'v_proj'],
    'gradient_accumulation_steps' : 2,
    'lora_num_epochs': 2,
    'lora_val_iterations': 100,
    'lora_early_stopping_patience': 10,
    'lora_lr': 1e-4,
    'lora_micro_batch_size': 1
}

# Alpaca-LoRA Instruction Templates
ALPACA_LORA_PROMPTS_USER_PROFILE = {
        'instruction': "### Instruction:\n You are a recommender system specialized in creating user profiles. Your task is to create a comprehensive user profile based on a user's review. The user Profile consists of: 1. Short-term Interests: Examine the user's most recent items along with their personalized descriptions 2. Long-term Preferences: Analyze all reviews and items from the user’s entire history to capture deeper, stable interests that define the user’s lasting preferences. Use this comprehensive historical data to outline consistent themes and inclinations that have remained steady over time. 3. User Profile Summary: Synthesize these findings into a concise profile (max 200 words) that combines insights from both short-term interests and long-term preferences. Use the short-term interests as a query to refine or highlight relevant themes from the long-term history, and provide a cohesive picture of the user’s tastes, typical habits, and potential future interests.Present your analysis under clear section headings.Do Not include any Code in your response. Think step by step",
        'input': "{user_review}",
        'output': "### Response:"
    }

ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS = {
        'instruction': "### Instruction:\n You are a recommender system specialized in creating user profiles. Your task is to create a comprehensive user profile and a list of candidate items Categories based on a user's review. The user Profile consists of: 1. Short-term Interests: Examine the user's most recent items along with their personalized descriptions 2. Long-term Preferences: Analyze all reviews and items from the user’s entire history to capture deeper, stable interests that define the user’s lasting preferences. Use this comprehensive historical data to outline consistent themes and inclinations that have remained steady over time. 3. User Profile Summary: Synthesize these findings into a concise profile (max 200 words) that combines insights from both short-term interests and long-term preferences. Use the short-term interests as a query to refine or highlight relevant themes from the long-term history, and provide a cohesive picture of the user’s tastes, typical habits, and potential future interests. Afterwards generate five Candidate Items that are general product categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction. ",
        'input': "User Reviews: {user_review}",
        'output': "### Response:"
    }
ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS = {
    "instruction":  "### Instruction:\n You are a recommender system specialized. Based on the following user profile text, generate a list of general candidate item categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction. ",
        'input': "### Input \n User Profile: \n {user_profile}",
        'output': "### Response:"
}
# Prompts

USER_PROFILE_AND_CANDIDATE_ITEM = """
You are a recommender system specialized in creating user profiles and generating Candidate Items Categories:.
Based on the following user reviews, listed in chronological order (oldest to newest):
{reviews}
**Task 1: User Profile Creation**
Analyze this information and create a user profile by following these steps:
1. **Short-term Interests:** Examine the user's most recent items along with their personalized descriptions.
2. **Long-term Preferences:** Analyze all reviews from the user's entire history to capture deeper, stable interests that define the user's lasting preferences. Outline consistent themes and inclinations that have remained steady over time.
3. **User Profile Summary:** Synthesize these findings into a concise profile (maximum 200 words) that combines insights from both short-term interests and long-term preferences. Provide a cohesive picture of the user's tastes, typical habits, and potential future interests.
**Task 2: Product Categories Generation**
Based on the user profile you've created, generate five general product categories that align with the user's preferences and interests. Treat these categories as a cohesive set to reflect the user's overall profile and maximize satisfaction.
- **Identify Unique Aspects:** Create five distinct categories that capture different aspects of the user's profile.
- **Rank by Relevance:** Order the categories by their relevance to the user's profile, from most to least relevant.#
Present your analysis under clear section headings.Do not include any code in your response.Present the product categories as a numbered list that only contains the category name.

Example Output:
User Profile:
1."Short-Term Interests": "The user recently reviewed products focused on facial skincare, specifically creams and serums targeting anti-aging concerns such as fine lines, wrinkles, and dehydration. They showed interest in natural and organic ingredients, vegan-friendly options, and cruelty-free practices. Their preference leans towards face washes and moisturizers that cater to specific skin concerns like acne-prone, sensitive, and dry skin.",
2."Long-term Preferences": "Based on the entirety of their reviews, some common denominators emerge:\n* Focus on natural ingredients, particularly plant-based extracts, essential oils, and herbal remedies\n* Concern for anti-aging and rejuvenation, seeking effective solutions for maintaining healthy-looking skin\n* Interest in sheet masks, especially those with unique features like eye patches and targeted application areas\n* Appreciation for moisturized and hydrated skin, often mentioning specific requirements like deep hydration and non-sticky textures\n* Willingness to explore various brands and products, demonstrating adaptability and open-mindedness",
3."User Profile Summary": "Our user appears to be someone who prioritizes natural and organic approaches to skincare while focusing on addressing specific skin concerns. They exhibit a willingness to experiment with diverse products and techniques to achieve optimal results. As they continue exploring the world of skincare, they may gravitate toward more niche markets, such as bespoke formulas tailored to individual skin types or advanced technology-driven treatments. For now, our recommendation would focus on recommending products featuring natural, sustainable, and innovative formulations catering to their varied skin needs, including hydration, anti-aging, and sensitivity management. A suggested next step could involve introducing them to emerging trends in customized skincare regimens and cutting-edge technologies for enhanced customer experience."
Candidate Items Categories:
1. Sustainable Beauty Products
2. Luxury Skincare Essentials
3. Everyday Makeup Basics
4. Seasonal Self-Care Kits
5. Innovative Hair Care Solutions
"""
# Prompts
USER_PROFILE_AND_CANDIDATE_ITEM_NO_UP = """
You are a recommender system specialized in generating product categories.
Based on the following user reviews, listed in chronological order (oldest to newest):
{reviews}
**Task 1: Candidate Items Categories**
Based on the user profile you've created, generate five general product categories that align with the user's preferences and interests. Treat these categories as a cohesive set to reflect the user's overall profile and maximize satisfaction.
- **Identify Unique Aspects:** Create five distinct categories that capture different aspects of the user's profile.
- **Rank by Relevance:** Order the categories by their relevance to the user's profile, from most to least relevant.

Present your analysis under clear section headings.Do not include any code in your response.Present the product categories as a numbered list that only contains the category name.
Candidate Items Categories:
1. Sustainable Beauty Products
2. Luxury Skincare Essentials
3. Everyday Makeup Basics
4. Seasonal Self-Care Kits
5. Innovative Hair Care Solutions
"""

USER_PROFILE_PROMPT = """
You are a recommender system specialized in creating user profiles. Your task is to create a comprehensive user profile based on the following reviews, listed in chronological order (oldest to newest):

{reviews}

Analyze this information and create a user profile following these steps:

1. Short-term Interests: Examine the user's most recent items along with their personalized descriptions
2. Long-term Preferences: Analyze all reviews and items from the user’s entire history to capture deeper, stable interests that define the user’s lasting preferences. Use this comprehensive historical data to outline consistent themes and inclinations that have remained steady over time.
3. User Profile Summary: Synthesize these findings into a concise profile (max 200 words) that combines insights from both short-term interests and long-term preferences. Use the short-term interests as a query to refine or highlight relevant themes from the long-term history, and provide a cohesive picture of the user’s tastes, typical habits, and potential future interests

Present your analysis under clear section headings. Ensure the final profile reflects both the user’s stable and current interests for a comprehensive understanding of their preferences.Do Not include any Code in your response"""

PRELIMINARY_RECOMMENDATIONS_WITH_PRODUCT_DESCRIPTION_PROMPT = """You are a recommender system specialized in generating product categories based on user profiles.

Based on the following user profile:

{user_profile}

Generate five general Candidate Items Categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction.

Instructions:

    Identify Unique Aspects: Create five distinct categories that capture different aspects of the user’s profile, aiming to reflect a broad range of their preferences and interests.
    Enhance Cohesion: Ensure that each category complements the others, creating a set that feels cohesive and well-rounded.
    Rank by Relevance: Order the categories by their relevance to the user’s profile, from most to least relevant, while ensuring each adds meaningful variety to the list.
Present the result as a numbered list. Do not include any code in your response.
Example Output:
Candidate Items Categories:
1. Sustainable Beauty Products – Reflecting the user’s long-standing preference for eco-friendly and organic items.
2. Luxury Skincare Essentials – Highlighting interest in premium skincare brands, often favoring high-quality, indulgent products.
3. Everyday Makeup Basics – Covering frequently purchased items for daily use, aligned with recent purchases of affordable, versatile products.
4. Seasonal Self-Care Kits – Capturing a trend of seasonally themed sets, ideal for users who enjoy holiday promotions and gift-ready collections.
5. Innovative Hair Care Solutions – Targeting a new interest in specialized hair treatments, noted in recent product searches and reviews.

"""
PRELIMINARY_RECOMMENDATIONS_PROMPT = """You are a recommender system specialized in generating product categories based on user profiles.
Based on the following user profile:
{user_profile}
Generate five Candidate Items Categories that align with the user's preferences and interests. Approach this task by treating these categories as a cohesive set, ensuring that they collectively reflect the user’s overall profile and maximize satisfaction.
Instructions:

    Identify Unique Aspects: Create five distinct categories that capture different aspects of the user’s profile, aiming to reflect a broad range of their preferences and interests.
    Enhance Cohesion: Ensure that each category complements the others, creating a set that feels cohesive and well-rounded.
    Rank by Relevance: Order the categories by their relevance to the user’s profile, from most to least relevant, while ensuring each adds meaningful variety to the list.

Present the result as a numbered list that only contains the category name. Do not include any code in your response.
Example Output:
Candidate Item Categories:
1. Sustainable Beauty Products 
2. Luxury Skincare Essentials 
3. Everyday Makeup Basics 
4. Seasonal Self-Care Kits 
5. Innovative Hair Care Solutions
"""

EMBEDDING_MODEL_CONFIG = {
    "name": "hyp1231/blair-roberta-large",
    "batch_size": 32,
    "max_length": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

CHROMA_CONFIG = {
    "base_path": "./chroma_db",
    "embedding_function": "default",
    "distance_function": "cosine"
}