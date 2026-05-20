# paperqa/scripts/01_prepare_introductions.py

import os
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERQA_DIR = PROJECT_ROOT / "data" / "paper_dataset"
SAVE_DIR = PROJECT_ROOT / "paperqa" / "data"
sys.path.append(str(PAPERQA_DIR))

def prepare_individual_introductions():
    """
    Extracts the introduction of each paper from paperQA.json and 
    creates individual files for each paper.
    
    Saves in two formatting styles:
    1. [Paper ID: {paper_id}] format
    2. This paper {paper_id} discusses format
    """
    # Input file path
    input_file = PAPERQA_DIR / "paperQA.json"
    
    # Output directories for the two formats
    output_dir_bracket = SAVE_DIR / "individual_introductions_bracket"
    output_dir_natural = SAVE_DIR / "individual_introductions_natural"
    os.makedirs(output_dir_bracket, exist_ok=True)
    os.makedirs(output_dir_natural, exist_ok=True)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"'{input_file}' not found.")
    
    logging.info(f"Extracting introduction data per paper from '{input_file}'.")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    

    for paper in papers:
        paper_id = paper['id']
        introduction = paper['introduction']
        
        # 1. [Paper ID: {paper_id}] format
        formatted_text_bracket = f"[Paper ID: {paper_id}]\n\n{introduction}"
        output_file_bracket = output_dir_bracket / f"paper_{paper_id}_introduction.jsonl"
        with open(output_file_bracket, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"text": formatted_text_bracket}, ensure_ascii=False) + '\n')
        
        # 2. This paper {paper_id} discusses format
        formatted_text_natural = f"This paper {paper_id} discusses:\n\n{introduction}"
        output_file_natural = output_dir_natural / f"paper_{paper_id}_introduction.jsonl"
        with open(output_file_natural, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"text": formatted_text_natural}, ensure_ascii=False) + '\n')
        
        logging.info(f"Saved introduction for paper {paper_id} in two formats.")
        logging.info(f"  - Bracket format: '{output_file_bracket}'")
        logging.info(f"  - Natural format: '{output_file_natural}'")
    
    logging.info(f"Generated introduction data for a total of {len(papers)} papers in two formats.")
    return papers

def prepare_concatenated_introductions(papers):
    """
    Extracts introductions from all papers and concatenates them 
    into a single file.
    
    Saves in two formatting styles:
    1. [Paper ID: {paper_id}] format
    2. This paper {paper_id} discusses format
    """
    # Output file paths
    output_file_bracket = SAVE_DIR / "concatenated_introductions_bracket.jsonl"
    output_file_natural = SAVE_DIR / "concatenated_introductions_natural.jsonl"
    
    logging.info("Concatenating introductions for all papers in two formats.")
    
    # Concatenate all introductions (two formats)
    concatenated_text_bracket = ""
    concatenated_text_natural = ""
    
    for paper in papers:
        paper_id = paper['id']
        introduction = paper['introduction']
        # 1. [Paper ID: {paper_id}] format
        formatted_section_bracket = f"[Paper ID: {paper_id}]\n\n{introduction}\n\n"
        concatenated_text_bracket += formatted_section_bracket
        
        # 2. This paper {paper_id} discusses format
        formatted_section_natural = f"This paper {paper_id} discusses:\n\n{introduction}\n\n"
        concatenated_text_natural += formatted_section_natural
    
    # Save files in two formats
    with open(output_file_bracket, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"text": concatenated_text_bracket}, ensure_ascii=False) + '\n')
    
    with open(output_file_natural, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"text": concatenated_text_natural}, ensure_ascii=False) + '\n')
    
    logging.info(f"Concatenated and saved introductions for a total of {len(papers)} papers in two formats.")
    logging.info(f"  - Bracket format: '{output_file_bracket}'")
    logging.info(f"  - Natural format: '{output_file_natural}'")

def main():
    """
    Generates both individual and concatenated introduction data.
    """
    logging.info("=== Starting Introduction Data Preparation ===")
    
    # 1. Generate individual introduction data
    papers = prepare_individual_introductions()
    
    # 2. Generate concatenated introduction data
    prepare_concatenated_introductions(papers)
    
    logging.info("=== Introduction Data Preparation Complete ===")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
