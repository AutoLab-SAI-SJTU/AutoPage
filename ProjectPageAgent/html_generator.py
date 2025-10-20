"""
HTML generator for project page generation.
Generates the final HTML project page from planned content.
"""

import json
import yaml
import os
import io
import re
import json
import yaml
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from jinja2 import Environment, StrictUndefined
from camel.models import ModelFactory
from camel.agents import ChatAgent
from utils.wei_utils import get_agent_config, account_token
from utils.src.utils import get_json_from_response, extract_html_code_block
from ProjectPageAgent.css_checker import check_css 
from utils.src.utils import run_sync_screenshots
from PIL import Image
from camel.messages import BaseMessage


from camel.models import ModelFactory

def to_url(input_path_or_url: str) -> str:
    parsed = urlparse(input_path_or_url)
    if parsed.scheme in ("http", "https", "file"):
        return input_path_or_url
    p = Path(input_path_or_url).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    return p.as_uri()  # file://...


def crop_image_to_max_size(image_path, max_bytes=8*1024*1024, output_path=None):
    img = Image.open(image_path)
    img_format = img.format
    if output_path is None:
        output_path = image_path

    buffer = io.BytesIO()
    img.save(buffer, format=img_format)
    size = buffer.getbuffer().nbytes

    if size <= max_bytes:
        img.save(output_path, format=img_format)
        return output_path

    width, height = img.size
    scale = max_bytes / size
    new_height = max(int(height * scale), 1)  
    img_cropped = img.crop((0, 0, width, new_height))  
    img_cropped.save(output_path, format=img_format)

    return output_path
class ProjectPageHTMLGenerator:
    """Generates HTML project pages from planned content."""
    
    def __init__(self, agent_config,args):
        self.agent_config = agent_config
        self.args = args
        self.html_agent = self._create_html_agent()
        self.review_agent = self._create_review_agent()
        self.table_agent = self._create_table_agent()
        self.long_agent = self._create_long_agent()
        
        # self.client = OpenAI(api_key=api_key,base_url=api_url)
        
    def _create_html_agent(self):
        """Create the HTML generation agent."""
        model_type = str(self.agent_config['model_type'])
        if model_type.startswith('vllm_qwen') or 'vllm' in model_type.lower():
            model = ModelFactory.create(
                model_platform=self.agent_config['model_platform'],
                model_type=self.agent_config['model_type'],
                model_config_dict=self.agent_config['model_config'],
                url=self.agent_config.get('url', None),
            )
        else:
            model = ModelFactory.create(
                model_platform=self.agent_config['model_platform'],
                model_type=self.agent_config['model_type'],
                model_config_dict=self.agent_config['model_config'],
            )
        
        system_message = """You are an expert web developer specializing in creating professional project pages for research papers. 
        You have extensive experience in HTML5, CSS3, responsive design, and academic content presentation. 
        Your goal is to create engaging, well-structured, and visually appealing project pages."""
        
        return ChatAgent(
            system_message=system_message,
            model=model,
            message_window_size=10
        )
    def _create_review_agent(self):
        with open('utils/prompt_templates/page_templates/html_review.yaml', 'r') as f:
            prompt_config = yaml.safe_load(f)

        jinja_env = Environment(undefined=StrictUndefined)
        system_message_template = jinja_env.from_string(prompt_config["system_prompt"])

        system_message = system_message_template.render()

        model_type = self.args.model_name_v
        config = get_agent_config(model_type)
        model = ModelFactory.create(
            model_platform=config['model_platform'],
            model_type=config['model_type'],
            model_config_dict=config['model_config'],
            url=config.get('url', None),
        )

        return ChatAgent(
            system_message=system_message, 
            model=model,
            message_window_size=10
        )


    def _create_table_agent(self):
        
        model_type = self.args.model_name_v
      
        vlm_config = get_agent_config(model_type)
        vlm_model = ModelFactory.create(
            model_platform=vlm_config['model_platform'],
            model_type=vlm_config['model_type'],
            model_config_dict=vlm_config['model_config'],
            url=vlm_config.get('url', None),
        )
        return ChatAgent(
            system_message=None,
            model=vlm_model,
            message_window_size=10,
        )
    def _create_long_agent(self):
        model_type = self.args.model_name_t
        long_config = get_agent_config(model_type)
        long_model = ModelFactory.create(
            model_platform=long_config['model_platform'],
            model_type=long_config['model_type'],
            model_config_dict=long_config['model_config'],
            url=long_config.get('url', None),
        )
       
        return ChatAgent(
            system_message=None,
            model=long_model,
            message_window_size=10,
            token_limit=long_config.get('token_limit', None)
        )
    def render_html_to_png(self, iter, html_content, project_output_dir) -> str:

        import time
        tmp_html = Path(project_output_dir) / f"index_iter{iter}.html"
        tmp_html.write_text(html_content, encoding="utf-8")
        url = tmp_html.resolve().as_uri()

        image_path = str(Path(project_output_dir) / f"page_iter{iter}.png")

        run_sync_screenshots(url, image_path)
        return image_path

    def get_revision_suggestions(self, image_path: str, html_path) -> str:
        
        def crop_image_max_width(img, max_width=1280):
            width, height = img.size
            if width > max_width:
                img = img.crop((0, 0, max_width, height))  # (left, top, right, bottom)
            return img
        img = Image.open(image_path)
        img = crop_image_max_width(img, max_width=1280)
        img.save(image_path,format='PNG')
        crop_image_to_max_size(image_path=image_path,output_path=image_path)
        img =Image.open(image_path)
        
        message = BaseMessage.make_user_message(
                role_name="User",
                content = '\nHere is the image of the generated project page.',
                image_list=[img]
        )
        response = self.review_agent.step(message)

        return get_json_from_response(response.msgs[0].content.strip())
    

    def modify_html_table(self, html_content: str,html_dir: str):

        
        in_tokens, out_tokens = 0, 0
        print("Starting table modification...")
        def replace_tables_in_html(html_content, table_html_map, paper_name):
  
            pattern = rf'<img[^>]*src="(assets/{paper_name}-table-\d+\.png)"[^>]*>'
            
            def repl(match):
                img_path = match.group(1)  # e.g. assets/MambaFusion-table-10.png
                if img_path in table_html_map:
                    return table_html_map[img_path]
                return match.group(0)  
            
            return re.sub(pattern, repl, html_content)

        # ============ step 1 extract table ============
        
        pattern = rf"assets/{self.args.paper_name}-table-\d+\.png"
        with open(os.path.join(self.args.output_dir,self.args.paper_name, html_dir,'index_no_modify_table.html'), 'r', encoding='utf-8') as f:
            html_content = f.read()
        matches = re.findall(pattern, html_content)

        if matches is None:
            print("No table images found, skipping modification.")
            return None, 0, 0
        
     
        model_type = self.args.model_name_v
        print(f"Starting table modification phase 1: Table Extraction with {model_type}...")
        
        with open('utils/prompt_templates/page_templates/extract_table.yaml', 'r') as f:
            table_extraction_config = yaml.safe_load(f)
        content = table_extraction_config["system_prompt"]

        init_message = BaseMessage.make_user_message(
            role_name="User",
            content=content
        )
        response = self.table_agent.step(init_message)
        in_tok , out_tok = account_token(response)
        in_tokens += in_tok
        out_tokens += out_tok
        # Step 2
        table_html_map = {}

        matches = list(set(matches))
        for match in matches:
            img_path =os.path.join(self.args.output_dir,self.args.paper_name, html_dir,match)
            print(f"Processing table image: {img_path}")
            img = Image.open(img_path)
            msg = BaseMessage.make_user_message(
                role_name="User",
                content=f'''Here is table image: {match}
            Please output its HTML table (<table>...</table>) with an inline <style>...</style> block.
            Only return pure HTML , nothing else.
            ''',
                image_list=[img]
            )
            response = self.table_agent.step(msg)
            in_tok , out_tok = account_token(response)
            in_tokens += in_tok
            out_tokens += out_tok
            print(f'in:{in_tok},out:{out_tok}')
            _output_html = response.msgs[0].content.strip()
            table_html_map[match] = _output_html
            tabel_dir = os.path.join(self.args.output_dir,self.args.paper_name, html_dir)
            os.makedirs(f'{tabel_dir}/table_html', exist_ok=True)
            
            with open(f'{tabel_dir}/table_html/{match.replace("/", "_")}.html', 'w', encoding='utf-8') as f:
                f.write(table_html_map[match])

        # ============ 阶段 2：HTML Merge ============
   
        self.table_agent.reset()
        img_path =os.path.join(self.args.output_dir,self.args.paper_name, html_dir,'page_final_no_modify_table.png')
        img = Image.open(img_path)
        with open('utils/prompt_templates/page_templates/color_suggestion.yaml','r') as f:
            prompt_config = yaml.safe_load(f)

        jinja_env = Environment(undefined=StrictUndefined)
        init_prompt_template = jinja_env.from_string(prompt_config["system_prompt"])

        init_prompt = init_prompt_template.render()

        msg = BaseMessage.make_user_message(
            role_name="User",
            content=init_prompt, 
            image_list=[img]
        )

        color_response = self.table_agent.step(msg)
        color_suggestion = color_response.msgs[0].content.strip()
        in_tok , out_tok = account_token(color_response)
        in_tokens += in_tok
        out_tokens += out_tok

      
        print(f"Starting table modification phase 2: HTML Merging with {model_type}...")
        

        tables_str = "\n\n".join(
            [f"Table extracted for {fname}:\n{html}" for fname, html in table_html_map.items()]
        )
        with open("utils/prompt_templates/page_templates/merge_html_table.yaml",'r') as f:
            prompt_config = yaml.safe_load(f)
        
        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(prompt_config["template"])

        jinja_args = {
            'html_content': html_content,
            'color_suggestion': color_suggestion,
            'tables_str': tables_str
        }

        prompt = template.render(**jinja_args)

        final_message = BaseMessage.make_user_message(
            role_name = "User",
            content = prompt
        )

        for i in range(3):
            self.long_agent.reset()
            response = self.long_agent.step(final_message)
            in_tok, out_tok = account_token(response)
            in_tokens += in_tok
            out_tokens += out_tok
            output_html = response.msgs[0].content.strip()
            print(f'in:{in_tok},out:{out_tok}')
            exteact_html_code = extract_html_code_block(output_html)
            if exteact_html_code is not None:
                break
            print(f"html format is not correct, regenerate {i} turn")
        
        return exteact_html_code, in_tokens, out_tokens


    def modify_html_from_human_feedback(self, html_content: str, user_feedback: str):
        """
        Modify HTML based on human feedback using the HTML agent.
        
        Args:
            html_content: Original HTML content
            user_feedback: Feedback from human reviewers
            
        Returns:
            str: Modified HTML content
        """
        in_tokens, out_tokens = 0, 0
        print("Starting HTML modification based on human feedback...")
        with open('utils/prompt_templates/page_templates/modify_html_from_human_feedback.yaml', 'r') as f:
            modifier_config = yaml.safe_load(f)

        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(modifier_config["template"])

        jinja_args = {
            'generated_html': html_content,
            'user_feedback': user_feedback
        }

        prompt = template.render(**jinja_args)
        for i in range(3):
            self.html_agent.reset()
            response = self.html_agent.step(prompt)
            in_tok, out_tok = account_token(response)
            in_tokens += in_tok
            out_tokens += out_tok
            print(f'input_token: {in_tok}, output_token: {out_tok}')
            modified_html = extract_html_code_block(response.msgs[0].content)

            if modified_html is not None:
                break
            print(f"html format is not correct, regenerate {i} turn")
        
        return modified_html, in_tokens, out_tokens
    

    def generate_complete_html(self, args, generated_content, html_dir, html_template=None):
        """
        Generate complete HTML by combining all sections, then render to PNG,
        send to OpenAI API for feedback, and regenerate HTML with suggestions.
        """
        
        # Create output directory for this specific project
        project_output_dir = f"{args.output_dir}/{args.paper_name}"
        html_path = os.path.join(project_output_dir, html_dir)
        if args.resume != 'html_check':
            with open('utils/prompt_templates/page_templates/html_generation.yaml', 'r') as f:
                generator_config = yaml.safe_load(f)

            jinja_env = Environment(undefined=StrictUndefined)
            template = jinja_env.from_string(generator_config["template"])

            jinja_args = {
                'generated_content': json.dumps(generated_content, indent=2),
                'html_template': html_template,
            }

            prompt = template.render(**jinja_args)
            for i in range(3):
                self.html_agent.reset()
                # print(self.html_agent)
                
                response = self.html_agent.step(prompt)
                # print(response.msgs[0].content)
                input_token, output_token = account_token(response)
                print(f'input_token: {input_token}, output_token: {output_token}')
                #print(input_token, output_token)
                html_content = extract_html_code_block(response.msgs[0].content)

                if html_content is not None:
                    break
                print(f"html format is not correct, regenerate {i} turn")
            

            # check css paths
            html_content = check_css(html_content, html_template)

            with open(os.path.join(html_path, 'index_init.html'),'w') as f:
                f.write(html_content)

            print(f"Initial HTML generation completed. Tokens: {input_token} -> {output_token}")

        else: 
            with open(os.path.join(html_path, 'index_init.html'), 'r', encoding='utf-8') as f:
                html_content = f.read()
        
        revised_html = html_content
        
        for i in range(self.args.html_check_times):
            if i==0:
                print("starting html check and revision...")
   
            image_path = self.render_html_to_png(i, revised_html, html_path)

            suggestions = self.get_revision_suggestions(image_path,os.path.join(html_path,f'index_iter{i}.html'))
            # print(f"Revision suggestions from {self.args.model_name_v}:\n", suggestions)
            
            review_path = f'project_contents/{args.paper_name}_html_review_iter{i}.json'
            with open(review_path, 'w') as f:
                json.dump(suggestions, f, indent=4)

            self.html_agent.reset()
            with open('utils/prompt_templates/page_templates/html_modify_from_suggestion.yaml', 'r') as f:
                regenerator_config = yaml.safe_load(f)

            jinja_env = Environment(undefined=StrictUndefined)
            _template = jinja_env.from_string(regenerator_config["template"])

            _jinja_args = {
                'existing_html': revised_html,
                'suggestions': suggestions
            }

            revision_prompt = _template.render(**_jinja_args)

            # print(revision_prompt)
            revised_response = self.html_agent.step(revision_prompt)
            # print(revised_response.msgs[0].content)
            revised_html = extract_html_code_block(revised_response.msgs[0].content)

            print("Revised HTML generation completed.")
            input_token, output_token = account_token(revised_response)
            print(f'in:{input_token}, out:{output_token}')

        return revised_html, input_token, output_token

    
    def save_html_file(self, html_content, args, html_dir, output_dir="generated_project_pages"):
        """
        Save the generated HTML to a file.
        
        Args:
            html_content: Generated HTML content
            args: Command line arguments
            output_dir: Output directory for the HTML file
            
        Returns:html_check
            str: Path to the saved HTML file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output directory for this specific project
        project_output_dir = f"{output_dir}/{args.paper_name}"
        os.makedirs(project_output_dir, exist_ok=True)
        
        # Save HTML file
        html_file_path = f"{project_output_dir}/{html_dir}/index.html"
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML project page saved to: {html_file_path}")
        
        return html_file_path
    
    def create_assets_directory(self, args, html_dir, output_dir="generated_project_pages"):
        """
        Create assets directory and copy only specific images/tables.

        This function creates an 'assets' subdirectory inside the project output path,
        then selectively copies images whose filenames contain '-picture-' or '-table-'.

        Args:
            args: Command line arguments (expects args.paper_name)
            html_dir (str): Directory name where HTML files are stored.
            output_dir (str): Base output directory. Default is "generated_project_pages".
        
        Returns:
            str: Path to the created assets directory.
        """
        import os
        import shutil

        # 构建输出路径
        project_output_dir = os.path.join(output_dir, args.paper_name)
        assets_dir = os.path.join(project_output_dir, html_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # 源图像与表格路径
        source_assets_dir = os.path.join("generated_project_pages", "images_and_tables", args.paper_name)

        # 检查源路径是否存在
        if os.path.exists(source_assets_dir):
            for file in os.listdir(source_assets_dir):
                # 只处理图片格式文件
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    continue

                # 文件名中必须包含 '-picture-' 或 '-table-'
                if "-picture-" in file or "-table-" in file:
                    src_path = os.path.join(source_assets_dir, file)
                    dst_path = os.path.join(assets_dir, file)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {file}")
                else:
                    print(f"Skipped (not picture/table): {file}")

        print(f"Assets directory created at: {assets_dir}")
        return assets_dir

    
    def generate_metadata(self, generated_content, args):
        """
        Generate metadata for the project page.
        
        Args:
            generated_content: Generated content
            args: Command line arguments
            
        Returns:
            dict: Metadata for the project page
        """
        metadata = {
            'title': generated_content.get('meta', {}).get('poster_title', 'Research Project'),
            'description': generated_content.get('meta', {}).get('abstract', '')[:160],
            'authors': generated_content.get('meta', {}).get('authors', ''),
            'affiliations': generated_content.get('meta', {}).get('affiliations', ''),
            'keywords': [],
            'generated_by': f"Paper2ProjectPage ({args.model_name_t}_{args.model_name_v})",
            'generation_date': str(datetime.now())
        }
        
        # Extract keywords from content
        content_text = json.dumps(generated_content, ensure_ascii=False)
        # Simple keyword extraction (can be improved)
        words = content_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 10 most frequent words as keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        metadata['keywords'] = [word for word, freq in sorted_words[:10]]
        
        return metadata
    
    def save_metadata(self, metadata, args, output_dir="generated_project_pages"):
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: Generated metadata
            args: Command line arguments
            output_dir: Output directory
            
        Returns:
            str: Path to the saved metadata file
        """
        project_output_dir = f"{output_dir}/{args.paper_name}"
        metadata_file_path = f"{project_output_dir}/metadata.json"
        
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        print(f"Metadata saved to: {metadata_file_path}")
        return metadata_file_path 