from .pdf_to_dict import retrieve_content_from_pdf
from .ppl_eval import get_full_content_ppl
from .question_generate import generate_question_answer
import os
from .question_answer import answer_question
import json
from .eval_qa import acc_qa
from .similar_score_eval import calculate_similar_score_ours
from .eval_compression import eval_compression
from pathlib import Path
from urllib.parse import urlparse
import argparse
import time
from utils.src.utils import run_sync_screenshots
from PageEval.vlm_judge import vlm_judge
from dotenv import load_dotenv
def to_url(input_path_or_url: str) -> str:
    parsed = urlparse(input_path_or_url)
    if parsed.scheme in ("http", "https", "file"):
        return input_path_or_url
    p = Path(input_path_or_url).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    return p.as_uri()  


load_dotenv()
def main():
    # 01-ai/Yi-6B
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run PageEval on a paper.")
    parser.add_argument("--paper_name", required=True, help="Paper name, e.g. test1")
    parser.add_argument("--html_path", required=True, help="HTML path")
    parser.add_argument("--model_name", default="01-ai/Yi-6B", help="Model name for perplexity evaluation")
    parser.add_argument("--vlm_name", default="gpt-4o",required=True, help="Model name for vlm")
    parser.add_argument("--llm_name", default="gpt-4o", required=True,help="Model name for llm")
    parser.add_argument("--stride", type=int, default=12, help="Stride size for perplexity evaluation")
    parser.add_argument('--output_dir', type=str, default='eval_result', help='Output directory for eval')
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)
    full_content_path = os.path.join('project_contents',f'{args.paper_name}_generated_full_content.json')
    raw_content_path = os.path.join('project_contents',f'{args.paper_name}_raw_content.json')
    output_dir  = os.path.join(args.output_dir,args.paper_name)
    os.makedirs(output_dir,exist_ok=True)
    question_path = os.path.join(output_dir,'question.json')
    result_output_path = os.path.join(output_dir,'result.json')
   
    os.makedirs('tmp',exist_ok=True)
    image_path = os.path.join('tmp',f'{args.paper_name}.png')
        
    # ppl
    result_ppl = get_full_content_ppl(full_content_path,args.model_name,args.stride)  
    
    # similar score
    extract_pdf_json = retrieve_content_from_pdf(full_content_path,raw_content_path,args.llm_name)
    result_similar = calculate_similar_score_ours(extract_pdf_json,full_content_path)
    avg_score = sum(result_similar.values()) / len(result_similar) if result_similar else 0
    result_similar['avg'] = avg_score
    
    # vlm_judge
    run_sync_screenshots(to_url(args.html_path),image_path)
    result_vlm_judge = vlm_judge(image_path,args.vlm_name)
    
    # compression
    result_com = eval_compression(
        html_full_content_json_path=full_content_path,
        pdf_raw_content_json_path= raw_content_path
    )
    
    # qa
    generate_question_answer(raw_content_path,question_path,'openrouter_openai/o3')
    answer_type = 'answer_question_from_text'
    answer_path = os.path.join(output_dir,f'{answer_type}.json')
    answer_question(
        answer_type= answer_type,
        html_path=full_content_path,
        vlm_type=args.vlm_name,
        llm_type=args.llm_name,
        output_path=answer_path,
        question_path=question_path
    )
    result_s_final = acc_qa(
        question_path=question_path,
        answer_path=answer_path,
        compression=result_com
    )
    
    result = {
        'ppl':result_ppl,
        'similar_score':result_similar,
        'qa':result_s_final,
        'compression':result_com,
        'vlm_judge':result_vlm_judge
    }
    
    with open(result_output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    end_time = time.time()
    print(f'use time:{end_time-start_time} s')
    os.remove(image_path)

# Usage example for simple version
if __name__ == "__main__":
    main()