import json
from transformers import AutoTokenizer
import os
import warnings

warnings.filterwarnings(
    "ignore", 
    message="Token indices sequence length is longer than the specified maximum"
)
import warnings
from transformers import AutoTokenizer
from transformers.utils import logging

def find_file(folder_path: str, file_name: str) -> str | None:
   
    folder_path = os.path.abspath(folder_path)

    if not os.path.isdir(folder_path):
        print(f"error: does not exist {folder_path} not exist")
        return None

    for root, dirs, files in os.walk(folder_path):
        if file_name in files:
            return os.path.join(root, file_name)
    

    print(f"error: Cant find {file_name} from{folder_path}")
    return None

def dict_to_text(data):
    text = ""
    for key, value in data.items():
        text += f"{key}:\n{value}\n\n"
    return text

from transformers import AutoTokenizer
import warnings

_tokenizer_cache = {}

def count_tokens(text, model_name='bert-base-uncased'):
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    logging.set_verbosity_error()  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def eval_text_compression(text1,text2):
    t1 = count_tokens(text1)
    t2 = count_tokens(text2)
    if t2==0:
        return 1000
    return t1/t2
    
def eval_compression(html_full_content_json_path,pdf_raw_content_json_path):
    with open(html_full_content_json_path,'r') as f :
        full_content = dict_to_text(json.load(f))
    with open(pdf_raw_content_json_path,'r') as f:
        raw_content = json.load(f)['markdown_content']
        
    return(eval_text_compression(raw_content,full_content))
# use example
if __name__ == "__main__":
    
    html_full_content_json_path = "project_contents/AutoPage_generated_full_content.json"
    pdf_raw_content_json_path = "project_contents/AutoPage_raw_content.json"
    with open(html_full_content_json_path,'r') as f :
        full_content = dict_to_text(json.load(f))
    with open(pdf_raw_content_json_path,'r') as f:
        raw_content = json.load(f)['markdown_content']
        
    print(eval_text_compression(raw_content,full_content))
    

