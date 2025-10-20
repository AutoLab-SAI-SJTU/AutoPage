import json
from openai import OpenAI
import re
from typing import Dict
import yaml
import jinja2
import os
from PIL import Image
from utils.wei_utils import get_agent_config
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.types import ModelPlatformType, ModelType
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import html

base_dir = os.path.dirname(__file__)

def fix_json_format(json_str):
  
    json_str = json_str.replace('\ufeff', '')

    json_str = html.unescape(json_str)

    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

    json_str = json_str.replace('"', '"').replace('"', '"')
    json_str = json_str.replace(''', "'").replace(''', "'")
    
    return json_str

def try_relaxed_json_parse(json_str):
    try:
        decoder = json.JSONDecoder(strict=False)
        return decoder.decode(json_str)
    except:
        pass
    try:
        import demjson3
        return demjson3.decode(json_str)
    except:
        pass

    return None

def extract_json(output_str):

    if output_str is None:
        print("❌ output is None")
        return None
        
    if not isinstance(output_str, str):
        print(f"❌ type error: {type(output_str)}, 值: {output_str}")
        return None
    
    if not output_str.strip():
        print("❌ output string is None")
        return None

    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{.*\}",  
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            json_str = match.group(1) if pattern.startswith("```") else match.group(0)
            json_str = json_str.strip()

            json_str = fix_json_format(json_str)
            
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                continue

    try:
        fixed_str = fix_json_format(output_str.strip())
        result = json.loads(fixed_str)
        return result
    except json.JSONDecodeError as e:
        print(f"❌ JSON extract error: {e}")
        print(f"error location: line {e.lineno}, column {e.colno}")
        print(f"raw content [0:500]: {output_str[:500]}")
        print(f"raw content [-500：]: {output_str[-500:]}")

        result = try_relaxed_json_parse(output_str)
        if result:
            print("✅ extract success")
        return result

def answer_question(
    answer_type: str,
    html_path: str,
    vlm_type: str,
    llm_type: str,
    output_path: str,
    question_path:str,
    img_path: str = '',
) -> Dict:
    """
    parameter:
        paper_name (str): pdf name
        answer_type(str): three types: 
            answer_question_from_image or
            answer_question_from_text_no_ref or
            answer_question_from_text  

    return:
        str: answer
    """
    print('-'*50)
    print(f"Answering questions, answer type: {answer_type} ...")

    image = ''
    if answer_type == 'answer_question_from_image':
        image = Image.open(img_path)
    
    in_tokens = 0
    out_tokens = 0
    
    def call(system_prompt: str, user_prompt: str, vlm_type,llm_type,image=None) -> tuple:
        
        if answer_type == 'answer_question_from_image':
            agent_config = get_agent_config(vlm_type)
            user_prompt = BaseMessage.make_user_message(
                role_name="User",
                content=user_prompt,
                image_list=[image]
            )
        else:
            agent_config = get_agent_config(llm_type)
        
        model_type = str(agent_config['model_type'])
        if model_type.startswith('vllm_qwen') or 'vllm' in model_type.lower():
            model = ModelFactory.create(
                model_platform=agent_config['model_platform'],
                model_type=agent_config['model_type'],
                model_config_dict=agent_config['model_config'],
                url=agent_config.get('url', None),
            )
        else:
            model = ModelFactory.create(
                model_platform=agent_config['model_platform'],
                model_type=agent_config['model_type'],
                model_config_dict=agent_config['model_config'],
            )
        
        agent = ChatAgent(
            system_message=system_prompt,
            model=model,
            message_window_size=10,
            token_limit=agent_config.get('token_limit', None)
        )
        response = agent.step(user_prompt)
        output_str = response.msgs[0].content

        output_json = extract_json(output_str)

        usage = response.info.get("usage", {})
        input_tokens = usage.get("prompt_tokens", "N/A")
        output_tokens = usage.get("completion_tokens", "N/A")

        return output_json, input_tokens, output_tokens

    system_prompt, user_prompt = _get_prompt_template(answer_type, question_path=question_path, html_path=html_path)
    
    print(f'[answer question] answering question type: {answer_type}, waiting ...')
    result_detail = None
    result_understanding = None
    
    if answer_type == 'answer_question_from_image':
        while result_detail is None:
            result_detail, in_token, out_token = call(system_prompt, user_prompt[0], vlm_type,llm_type,image)
            in_tokens += in_token
            out_tokens += out_token
        while result_understanding is None:
            result_understanding, in_token, out_token = call(system_prompt, user_prompt[1], vlm_type,llm_type,image)
            in_tokens += in_token
            out_tokens += out_token
    else:
        while result_detail is None:
            result_detail, in_token, out_token = call(system_prompt, user_prompt[0],vlm_type,llm_type)
            in_tokens += in_token
            out_tokens += out_token
        while result_understanding is None:
            result_understanding, in_token, out_token = call(system_prompt, user_prompt[1],vlm_type,llm_type)
            in_tokens += in_token
            out_tokens += out_token
    
    result = {
        "detail": result_detail,
        "understanding": result_understanding
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f'input tokens: {in_tokens}')
    print(f'output tokens: {out_tokens}')

def _get_prompt_template(answer_type: str, html_path: str,question_path: str) -> str:

    template_path = os.path.join('utils', 'prompt_templates', 'page_templates', f'{answer_type}.yaml')
    with open(template_path, 'r', encoding='utf-8') as file:
        template = yaml.safe_load(file)
    with open(html_path,'r') as f:
        html_text = str(json.load(f))
        
    system_prompt = template['system_prompt'] 
    user_prompt = template['template'] 
    
   
    with open(question_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    questions1 = qa_data["detail"]["questions"]
    questions2 = qa_data["understanding"]["questions"]

    template_engine = jinja2.Template(user_prompt)

    if answer_type == "answer_question_from_text":
        user_prompt1 = template_engine.render(questions=questions1, html_text=html_text)
        user_prompt2 = template_engine.render(questions=questions2, html_text=html_text)
    elif answer_type == "answer_question_from_text_no_ref":
        user_prompt1 = template_engine.render(questions=questions1, html_text=html_text)
        user_prompt2 = template_engine.render(questions=questions2, html_text=html_text)
    else:
        user_prompt1 = template_engine.render(questions=questions1)
        user_prompt2 = template_engine.render(questions=questions2)

    return system_prompt, [user_prompt1, user_prompt2]

#use example
if __name__ == "__main__":
    
    answer_type = 'answer_question_from_text'
    html_full_content_json_path = 'project_contents/AutoPage_generated_full_content.json'
    question_path = 'qa_example.json'
    vlm_type = 'openrouter_gemini-2.5-flash'
    llm_type = 'openrouter_gemini-2.5-flash'
    output_path = 'answer.json'
    answer_question(
        answer_type,
        html_full_content_json_path,
        vlm_type,
        llm_type,
        output_path,
        question_path
    )
    