"""
Paper parsing module for ProjectPageAgent.
Reuses the parsing capabilities from Paper2Poster.
"""

from ProjectPageAgent.parse_raw import parse_raw, gen_image_and_table
from utils.wei_utils import get_agent_config
import json
import os
import argparse

def parse_paper_for_project_page(args, agent_config_t, version=2):
    """
    Parse a research paper PDF and extract content for project page generation.
    
    Args:
        args: Command line arguments
        agent_config_t: Text model configuration
        version: Parser version to use
    
    Returns:
        tuple: (input_tokens, output_tokens, raw_result, images, tables)
    """
    print("Step 1: Parsing the research paper...")
    
    # Add poster_path and poster_name attributes to args for compatibility with parse_raw
    if not hasattr(args, 'poster_path'):
        args.poster_path = args.paper_path
    
    if not hasattr(args, 'poster_name'):
        args.poster_name = args.paper_name
    
    # Parse the raw paper content
    input_token, output_token, raw_result = parse_raw(args, agent_config_t, version=version)
    
    # Extract images and tables
    _, _, images, tables = gen_image_and_table(args, raw_result)
    
    print(f"Parsing completed. Tokens: {input_token} -> {output_token}")
    print(f"Extracted {len(images)} images and {len(tables)} tables")
    
    return input_token, output_token, raw_result, images, tables

def save_parsed_content(args, raw_result, images, tables, input_token, output_token):
    """
    Save parsed content to files for later use.
    
    Args:
        args: Command line arguments
        raw_result: Parsed raw content
        images: Extracted images
        tables: Extracted tables
        input_token: Input token count
        output_token: Output token count
    """
    # Save raw content
    os.makedirs('project_contents', exist_ok=True)
    raw_content_path = f'project_contents/{args.paper_name}_raw_content.json'
    
    # Convert raw_result to JSON format if needed
    if hasattr(raw_result, 'document'):
        # Extract text content from docling result
        raw_markdown = raw_result.document.export_to_markdown()
        content_json = {
            'markdown_content': raw_markdown,
            'images': images,
            'tables': tables
        }
    else:
        content_json = raw_result
    
    with open(raw_content_path, 'w') as f:
        json.dump(content_json, f, indent=4)
    
    # Save token usage
    token_log = {
        'parse_input_tokens': input_token,
        'parse_output_tokens': output_token,
        'total_images': len(images),
        'total_tables': len(tables)
    }
    
    token_log_path = f'project_contents/{args.paper_name}_parse_log.json'
    with open(token_log_path, 'w') as f:
        json.dump(token_log, f, indent=4)
    
    print(f"Parsed content saved to {raw_content_path}")
    return raw_content_path, token_log_path 