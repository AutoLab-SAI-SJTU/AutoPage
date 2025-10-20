import json
import yaml
import os
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from utils.wei_utils import get_agent_config, account_token
import io
from PIL import Image
from PageEval.question_answer import extract_json
from utils.src.utils import get_json_from_response

def compress_image_to_max_size(image_path: str, max_size_mb: float = 9.0, output_path: str = None):
    
    if output_path is None:
        output_path = image_path

    max_bytes = max_size_mb * 1024 * 1024

    img = Image.open(image_path)
    img_format = img.format  

    with io.BytesIO() as buffer:
        img.save(buffer, format=img_format)
        orig_size = buffer.tell()

    if orig_size <= max_bytes:
        print(f"image size {orig_size/1024/1024:.2f} MB ≤ {max_size_mb} MB, no compress need")
        return image_path

    scale = (max_bytes / orig_size) ** 0.5  
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)

    print(f"raw image size: {orig_size/1024/1024:.2f} MB -> slide to {new_width}x{new_height}")

    img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

    img_resized.save(output_path, format=img_format)
    print(f"save to: {output_path}")
    return output_path


def vlm_judge(image_path,vlm_model):

    img_path = compress_image_to_max_size(image_path, 9, f'tmp_{os.getpid()}_{id(image_path)}.png')
    img = Image.open(img_path)
    agent_config = get_agent_config(vlm_model)
    model = ModelFactory.create(
            model_platform=agent_config['model_platform'],
            model_type=agent_config['model_type'],
            model_config_dict=agent_config['model_config'],
            url=agent_config.get('url', None),
        )
    judge_list = [
        'vlm_aesthetics_judge',
        'vlm_element_judge',
        'vlm_layout_judge'
    ]

    scores = {}
    for judge in judge_list:
        scores[judge] = 0
        
    for judge in judge_list:
        with open(f'utils/prompt_templates/page_templates/{judge}.yaml', 'r') as f:
            template_config = yaml.safe_load(f)
        system_prompt = template_config['system_prompt']
        template_prompt = template_config['template']
        
        agent = ChatAgent(
            system_message=system_prompt,
            model=model,
            message_window_size=10,
        )
        message = BaseMessage.make_user_message(
            role_name="User",
            content=template_prompt,
            image_list=[img]
        )
        response = agent.step(message)
        output = response.msgs[0].content
        print(output)
        output = get_json_from_response(output)
        print(output)
        ins, outs = account_token(response)
        print(ins, outs)
        sc = int(output['score'])
        print(sc)
        scores[judge] = sc
    
    return scores

# Usage example for simple version
if __name__=='__main__':
    image_path = 'generated_project_pages/AutoPage1/page_final.png'
    vlm_model = 'openrouter_gemini-2.5-flash'
    print(vlm_judge(image_path,vlm_model))