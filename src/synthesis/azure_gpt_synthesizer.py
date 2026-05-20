# src/synthesis/azure_gpt_synthesizer.py

import os
import json
import logging
from typing import List, Dict, Optional
from openai import AzureOpenAI
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Gpt4Synthesizer:
    def __init__(self):
        self.api_key = os.environ.get("AZURE_OPENAI_KEY")
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = "2024-10-21"
        self.deployment = "gpt-4.1"

        if not self.api_key:
            raise ValueError("The AZURE_OPENAI_KEY environment variable is not set.")
        if not self.azure_endpoint:
            raise ValueError("The AZURE_OPENAI_ENDPOINT environment variable is not set.")

        try:
            self.client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key
            )
            logging.info(f"AzureOpenAI client initialization complete. Deployment: '{self.deployment}'")
        except Exception as e:
            logging.error(f"Failed to initialize AzureOpenAI client: {e}")
            raise

    def _get_eval_qa_prompt(self, text_chunk: str, num_questions: int) -> str:
        return f"""
        You are an expert AI assistant tasked with creating a high-quality evaluation set for a language model.
        Based on the following research paper introduction, please generate exactly {num_questions} challenging question-answer pairs that test a deep understanding of the text.

        RULES:
        1. Questions should not be simple, surface-level queries. They should require reasoning, comparison, or synthesis of information presented in the text.
        2. Answers must be derived strictly from the provided text and be concise and accurate.
        3. Format your entire output as a single JSON object containing a list of dictionaries. Do not include any text outside of this JSON object.

        Each dictionary in the list should have two keys: "question" and "answer".

        EXAMPLE OUTPUT FORMAT:
        {{
            "qa_pairs": [
                {{
                    "question": "What is the primary limitation of traditional MCTS that MCTD aims to address?",
                    "answer": "Traditional MCTS relies on a forward model for tree rollouts, inheriting its limitations such as losing global consistency and being restricted to discrete action spaces."
                }}
            ]
        }}

        Now, please generate {num_questions} question-answer pairs based on the text below.

        TEXT:
        ---
        {text_chunk}
        ---
        """

    def _parse_response(self, response_content: str) -> List[Dict[str, str]]:
        try:
            json_str = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_str:
                data = json.loads(json_str.group(0))
                return data.get("qa_pairs", [])
            return []
        except json.JSONDecodeError:
            # Raise the error here to be caught by the outer except block
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during parsing: {e}")
            return []

    def generate_evaluation_qa(self, text_chunk: str, paper_id: str, num_questions: int = 100) -> Optional[List[Dict[str, str]]]:
        prompt = self._get_eval_qa_prompt(text_chunk, num_questions)
        
        # Setup path for saving raw responses
        raw_response_dir = "data/raw_responses"
        os.makedirs(raw_response_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_response_path = os.path.join(raw_response_dir, f"{paper_id}_{timestamp}_raw.txt")
        
        try:
            logging.info(f"Starting QA generation request to '{self.deployment}'... (Target: {num_questions})")
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=32768,
                top_p=1.0,
            )
            content = response.choices[0].message.content
            
            # Save raw response before parsing
            with open(raw_response_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Saved raw API response to '{raw_response_path}'.")
            
            parsed_qa = self._parse_response(content)
            
            if not parsed_qa and content: # Parsing failed but content exists
                logging.error("Failed to parse QA pairs. Please check the saved raw response file.")
                return None
            
            logging.info(f"Successfully generated and parsed {len(parsed_qa)} QA pairs.")
            return parsed_qa

        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from response. The raw response has been saved to '{raw_response_path}'. Please check the file content.")
            return None
        except Exception as e:
            logging.error(f"An error occurred during the Azure OpenAI API call: {e}")
            return None
