import json
import re
from typing import Dict
import yaml
import os
import random
from utils.wei_utils import get_agent_config
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.agents import ChatAgent
from utils.wei_utils import get_agent_config
import jinja2
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent


def extract_json(output_str):
    # extract ```json ... ``` or ``` ... ``` content
    match = re.search(r"```json(.*?)```", output_str, re.S)
    if not match:
        match = re.search(r"```(.*?)```", output_str, re.S)
    if match:
        output_str = match.group(1).strip()
    # try extract
    try:
        return json.loads(output_str)
    except json.JSONDecodeError as e:
        print("❌ JSON extract error:", e)
        print("raw return:", output_str[:500]) 
        return None

def get_answers_and_remove_answers(questions):
    question_only, answers, aspects = {}, {}, {}
    for key, val in questions.items():
        question_only[key] = {
            'question': val['question'],
            'options': val['options']
        }
        answers[key] = val['answer']
        aspects[key] = val['aspect']
    return question_only, answers, aspects

def generate_question_answer(
    pdf_path: str,
    output_path: str, 
    model_type: str,
) -> Dict:
   
    """
    extract content from pdf,and generate question and answer
    """
    in_tokens = 0
    out_tokens = 0
    print('-'*50)
    print(f"generate question and answer from pdf raw content...")
    with open(pdf_path,'r') as f:
        pdf_content = json.load(f)['markdown_content']
    def call_o3(system_prompt: str, user_prompt: str) -> tuple:
        agent_config = get_agent_config(model_type)
        model = ModelFactory.create(
                model_platform=agent_config['model_platform'],
                model_type=agent_config['model_type'],
                model_config_dict=agent_config['model_config'],
                url=agent_config.get('url', None),
            )
        agent = ChatAgent(
            system_message=system_prompt,
            model=model,
            message_window_size=10,
        )
        response = agent.step(user_prompt)
        output_str = response.msgs[0].content
        output_json = extract_json(output_str)
        usage = response.info.get("usage", {})
        input_tokens = usage.get("prompt_tokens", "N/A")
        output_tokens = usage.get("completion_tokens", "N/A")
        return output_json, input_tokens, output_tokens

    document_markdown = f"""
    PDF Content:
    {pdf_content}
    """

    # -------------------------------
    # Detail  prompt
    # -------------------------------
    with open(f"utils/prompt_templates/page_templates/generate_question_detail.yaml", "r", encoding="utf-8") as f:
        prompt_detail = yaml.safe_load(f)

    detail_sys_prompt = prompt_detail["system_prompt"]
    detail_template = prompt_detail["template"]

    template_engine = jinja2.Template(detail_template)
    detail_user_prompt = template_engine.render(document_markdown=document_markdown)

    detail_qa,in_token,out_token= call_o3(detail_sys_prompt, detail_user_prompt)
    in_tokens += in_token
    out_tokens += out_token
    # -------------------------------
    # Understanding  prompt
    # -------------------------------
    with open(f"utils/prompt_templates/page_templates/generate_question_understanding.yaml", "r", encoding="utf-8") as f:
        prompt_understanding = yaml.safe_load(f)

    understanding_sys_prompt = prompt_understanding["system_prompt"]
    understanding_template = prompt_understanding["template"]

    template_engine = jinja2.Template(understanding_template)
    understanding_user_prompt = template_engine.render(document_markdown=document_markdown)

    understanding_qa ,in_token,outs_token= call_o3(understanding_sys_prompt, understanding_user_prompt)
    in_tokens += in_token
    out_tokens += out_token
   
   
    detail_q, detail_a, detail_aspects = get_answers_and_remove_answers(detail_qa)
    understanding_q, understanding_a, understanding_aspects = get_answers_and_remove_answers(understanding_qa)

    final_qa = {}
    detail_qa = {
        'questions': detail_q,  
        'answers': detail_a,  
        'aspects': detail_aspects,  
    }

    understanding_qa = {
        'questions': understanding_q,  
        'answers': understanding_a,  
        'aspects': understanding_aspects,  
    }

    final_qa['detail'] = detail_qa  
    final_qa['understanding'] = understanding_qa 
    random_options(final_qa)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_qa, f, indent=4, ensure_ascii=False)
    print(f'input tokens: {in_tokens}')
    print(f'output tokens: {out_tokens}')

def random_options(qa: dict ) -> dict:
    
    options_letters = ['A', 'B', 'C', 'D']

    for i in range(50):
        question_key = f'Question {i+1}'

        if question_key not in qa['detail']['questions']:
            continue

        options_full = qa['detail']['questions'][question_key]['options']

        options_text = [opt.split('. ', 1)[1] for opt in options_full]

        choice_letter = random.choice(options_letters)
        index = ord(choice_letter) - ord('A')

        true_choice = qa['detail']['answers'][question_key][0]
        true_index = ord(true_choice) - ord('A')

        if index != true_index:
            options_text[true_index], options_text[index] = options_text[index], options_text[true_index]

        new_options = [f"{chr(ord('A') + idx)}. {text}" for idx, text in enumerate(options_text)]

        qa['detail']['questions'][question_key]['options'] = new_options

        qa['detail']['answers'][question_key] = qa['detail']['questions'][question_key]['options'][index]
    
    for i in range(50):
        question_key = f'Question {i+1}'

        if question_key not in qa['understanding']['questions']:
            continue

        options_full = qa['understanding']['questions'][question_key]['options']

        options_text = [opt.split('. ', 1)[1] for opt in options_full]

        choice_letter = random.choice(options_letters)
        index = ord(choice_letter) - ord('A')

        true_choice = qa['understanding']['answers'][question_key][0]
        true_index = ord(true_choice) - ord('A')

        if index != true_index:
            options_text[true_index], options_text[index] = options_text[index], options_text[true_index]

        new_options = [f"{chr(ord('A') + idx)}. {text}" for idx, text in enumerate(options_text)]

        qa['understanding']['questions'][question_key]['options'] = new_options
        qa['understanding']['answers'][question_key] = qa['understanding']['questions'][question_key]['options'][index]




# use example
if __name__ == "__main__":
    
    pdf_raw_content_json_path = "project_contents/AutoPage_raw_content.json"
    
    generate_question_answer(pdf_raw_content_json_path,output_path='qa_example.json')