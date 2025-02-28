# model_pipeline.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import *
import torch
from peft import PeftModel

class RecommenderModel:
    def __init__(self, sample_size=None, model_type='user_profile', adapter=False, 
                 dataset='beauty', data_source='chatgpt_data'):
        """
        Initializes the RecommenderModel with the given parameters.

        Args:
            sample_size (int or None): If specified, implies we want to use adapters (or a certain fine-tuned model).
            model_type (str): 'user_profile', 'both', etc. (existing logic).
            adapter (bool): Whether to use LoRA adapters or not.
            dataset (str): 'beauty' or 'video_games'
            data_source (str): 'chatgpt_data' or 'pipeline_data'
        """

        # Dynamically get model and tokenizer paths based on sample size and model_type
        if adapter and sample_size is not None:
            # 1) Load base model + tokenizer
            model_path = MODEL_PATH
            tokenizer_path = TOKENIZER_PATH
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # 2) Load adapters for user profile & candidate items 
            #    with new helper functions (dataset + data_source)
            self.adapter_path_user_profile = get_adapter_name_user_profile(sample_size, dataset, data_source)
            self.adapter_name_user_profile = "user_profile"
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.adapter_path_user_profile, 
                adapter_name=self.adapter_name_user_profile
            )
            
            self.adapter_path_candidate_items = get_adapter_name_candidate_items(sample_size, dataset, data_source)
            self.adapter_name_candidate_items = "candidate_items"
            self.model.load_adapter(
                self.adapter_path_candidate_items, 
                adapter_name=self.adapter_name_candidate_items
            )
            
            print(f"Loaded adapters: {list(self.model.peft_config.keys())}")
        else:
            # Fallback to base model
            model_path = MODEL_PATH
            tokenizer_path = TOKENIZER_PATH
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Initialize the pipeline
        self.pipeline = self.initialize_pipeline()

        # Store adapter flag
        self.adapter = adapter
        # Store dataset and data_source for reference (optional)
        self.dataset = dataset
        self.data_source = data_source

    def initialize_pipeline(self):
        """
        Initializes the text-generation pipeline.
        """
        # Add pad_token_id to PIPELINE_PARAMS
        pipeline_params = PIPELINE_PARAMS.copy()
        pipeline_params['pad_token_id'] = self.tokenizer.eos_token_id
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.eos_token_id,
            device_map="auto",
        )
        return pipe

    def get_response(self, prompt):
        """
        Generates a response from the model based on the input prompt.
        """
        sequences = self.pipeline(
            prompt,
            max_length=PIPELINE_PARAMS['max_length'],
            num_return_sequences=PIPELINE_PARAMS['num_return_sequences'],
            temperature=PIPELINE_PARAMS['temperature'],
            top_k=PIPELINE_PARAMS['top_k'],
            top_p=PIPELINE_PARAMS['top_p'],
            repetition_penalty=PIPELINE_PARAMS['repetition_penalty'],
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True,
        )
        gen_text = sequences[0]["generated_text"]
        # Remove the input prompt from the generated text
        response = gen_text[len(prompt):].strip()
        return response

    def create_user_profile(self, reviews, use_adapter):
        """
        Creates a user profile based on the provided reviews (without product descriptions).
        
        Args:
            reviews (str): The user's reviews.
            use_adapter (bool): Whether to use adapters.
        
        Returns:
            str: The generated user profile.
        """
        if use_adapter and self.adapter:
            # Use adapter-specific method
            print(f"\nSetting active adapter to: {self.adapter_name_user_profile}")
            self.model.set_adapter(self.adapter_name_user_profile)
            print(f"Current active adapter: {self.model.active_adapter}")
            prompt = (
                ALPACA_LORA_PROMPTS_USER_PROFILE['instruction'] + "\n\n" +
                ALPACA_LORA_PROMPTS_USER_PROFILE['input'].replace("{user_review}", reviews) + "\n" +
                ALPACA_LORA_PROMPTS_USER_PROFILE['output']
            )
        else:
            # Use the standard user profile prompt
            prompt = USER_PROFILE_PROMPT.format(reviews=reviews)
        return self.get_response(prompt)

    def create_preliminary_recommendations(self, user_profile, use_adapter, product_descriptions=False):
        """
        Generates preliminary recommendations based on the user profile.
        
        Args:
            user_profile (str): The user's profile.
            use_adapter (bool): Whether to use adapters.
            product_descriptions (bool): Whether to include product descriptions in the prompt.
        
        Returns:
            str: The generated preliminary recommendations.
        """
        if use_adapter and self.adapter:
            print(f"\nSetting active adapter to: {self.adapter_name_candidate_items}")
            self.model.set_adapter(self.adapter_name_candidate_items)
            print(f"Current active adapter: {self.model.active_adapter}")
            # Use the adapter-specific prompt
            prompt = (
                ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS['instruction'] + "\n\n" +
                ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS['input'].replace("{user_profile}", user_profile) + "\n" +
                ALPACA_LORA_PROMPTS_CANDIDATE_ITEMS['output']
            )
        elif product_descriptions:
            # Use the preliminary recommendations prompt with product descriptions
            prompt = PRELIMINARY_RECOMMENDATIONS_WITH_PRODUCT_DESCRIPTION_PROMPT.format(user_profile=user_profile)
        else:
            # Use the standard preliminary recommendations prompt
            prompt = PRELIMINARY_RECOMMENDATIONS_PROMPT.format(user_profile=user_profile)
        print(f"prelin promt {prompt}")
        return self.get_response(prompt)
        
    def create_user_profile_and_preliminary_recommendations(self, reviews, use_adapter):
            """
            Creates a user profile and candidate items based on the provided reviews using the combined prompt.

            Args:
                reviews (str): The user's reviews.
                use_adapter (bool): Whether to use adapters.

            Returns:
                str: The combined output containing both the user profile and candidate items.
            """
            if use_adapter:
                # Use adapter-specific method with your ALPACA prompt
                print("using fine tuned model")
                prompt = (
                    ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS['instruction'] + "\n\n" +
                    ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS['input'].replace("{user_review}", reviews) + "\n" +
                    ALPACA_LORA_PROMPTS_USER_PROFILE_AND_CANDIDATE_ITEMS['output']
                )
            else:
                # Use the standard combined prompt
                prompt = USER_PROFILE_AND_CANDIDATE_ITEM.format(reviews=reviews)
            return self.get_response(prompt)