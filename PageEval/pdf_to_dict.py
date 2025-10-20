import json
import re
import os
import json
from utils.wei_utils import get_agent_config
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent

def extract_json(output_str):

    match = re.search(r"```json(.*?)```", output_str, re.S)
    if not match:
        match = re.search(r"```(.*?)```", output_str, re.S)
    if match:
        output_str = match.group(1).strip()

    try:
        return json.loads(output_str)
    except json.JSONDecodeError as e:
        print("❌ JSON extract error:", e)
        print("raw return:", output_str[:500])  # 打印前 500 字排查
        return None

def remove_references(text: str) -> str:
    # remove references
    return re.split(r'(?i)\breferences\b', text, maxsplit=1)[0].strip()

def retrieve_content_from_pdf(html_full_content_json_path, pdf_raw_content_json_path, model_type):

    with open(html_full_content_json_path,'r') as f :
        html_sections = json.load(f)
        
    with open(pdf_raw_content_json_path,'r') as f:
        pdf_content = json.load(f)['markdown_content']
    def call_batch(section_keys, full_text,model_type):
        """
        一次性调用 Qwen-long API，批量提取多个 section
        """
        # 将所有 section keys 拼成提示词列表
        keys_list = "\n".join([f"- {key}" for key in section_keys])
        
        # 构造提示词
        prompt = f"""
        You are given the full text of a research paper and a list of target section keys.
        Your task is to extract the content of each target section as faithfully as possible from the full text, following these rules:
        1. For each section key, find the corresponding section by exact or close matching of section titles (e.g., "Results" can match "Experimental Results", "Quantitative Results", etc.).
        2. If the section content is clearly grouped under a heading, extract that content directly. If relevant content is scattered, you may combine or slightly summarize it, but do NOT invent or add any new content.
        3. Each extracted section must include all important technical details, methods, parameters, examples, results, data, and explanations.
        4. Skip any section keys related to "title", "authors", or "affiliation". Do NOT include them in the output.
        5. If no relevant content is found for a section key, omit it from the output.
        6. Return a single pure JSON string object, where keys are the section keys you received, and values are the extracted contents.
        Example format:

        {{
        "section_key1": "extracted content",
        "section_key2": "another extracted content"
        }}

        Target section keys:
        {keys_list}

        Full text:
        {full_text}

        """
        agent_config = get_agent_config(model_type)
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
            system_message="You are a precise document analysis assistant. Extract the requested sections accurately from the provided text content and return them as valid JSON.",
            model=model,
            message_window_size=10,
        )
        response = agent.step(prompt)
        output_str = response.msgs[0].content
        output_json = extract_json(output_str)

        usage = response.info.get("usage", {})
        input_tokens = usage.get("prompt_tokens", "N/A")
        output_tokens = usage.get("completion_tokens", "N/A")

        return output_json, input_tokens, output_tokens

    section_keys = list(html_sections.keys())
    print(f"Extracting section contents from pdf raw content  ...")

    batch_result ,in_tokens,out_tokens= call_batch(section_keys, pdf_content,model_type)
    
    print(f'input tokens: {in_tokens}')
    print(f'output tokens: {out_tokens}')
    if batch_result and isinstance(batch_result, dict):
        result_dict = {}
        for key in section_keys:
            if key in batch_result:
                result_dict[key] = batch_result[key]
            else:
                result_dict[key] = f"Content not found for: {key}"
        return result_dict
    else:
        return {key: f"Failed to retrieve content for: {key}" for key in section_keys}
    

# use example
if __name__ =='__main__':
    
    html_full_content_json_path = "project_contents/AutoPage_generated_full_content.json"
    pdf_raw_content_json_path = "project_contents/AutoPage_raw_content.json"
    
        
    print(retrieve_content_from_pdf(html_full_content_json_path,pdf_raw_content_json_path,model_type='openrouter_qwen-plus'))
    