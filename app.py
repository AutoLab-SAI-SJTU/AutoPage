import gradio as gr
import os
import json
from pathlib import Path
import base64
import re
from threading import Thread
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
from dotenv import load_dotenv
from ProjectPageAgent.parse_paper import parse_paper_for_project_page, save_parsed_content
from ProjectPageAgent.html_finder import HtmlFinder
from ProjectPageAgent.content_planner import ProjectPageContentPlanner
from ProjectPageAgent.html_generator import ProjectPageHTMLGenerator, to_url
from utils.wei_utils import get_agent_config
import os
import subprocess

from ProjectPageAgent.content_planner import filter_references
from utils.src.utils import run_sync_screenshots
from ProjectPageAgent.main_pipline import matching, copy_static_files

load_dotenv()



def get_agent_config_with_keys(model_type, openai_api_key="", gemini_api_key="", 
                              qwen_api_key="", zhipuai_api_key="", openrouter_api_key=""):
    """
    Get agent configuration with user-provided API keys.
    Falls back to environment variables if user keys are not provided.
    Note: This function sets environment variables but does NOT restore them.
    The environment variables will remain set for the duration of the application.
    """
    # Set environment variables with user-provided keys
    api_keys = {
        'OPENAI_API_KEY': openai_api_key,
        'GEMINI_API_KEY': gemini_api_key, 
        'QWEN_API_KEY': qwen_api_key,
        'ZHIPUAI_API_KEY': zhipuai_api_key,
        'OPENROUTER_API_KEY': openrouter_api_key
    }
    
    # Set new API keys in environment
    for key, value in api_keys.items():
        if value and value.strip():
            os.environ[key] = value
    
    # Get agent config with the new API keys
    config = get_agent_config(model_type)
    return config

def validate_api_keys(model_name_t, model_name_v, openai_api_key, gemini_api_key, 
                     qwen_api_key, zhipuai_api_key, openrouter_api_key):
    """
    Validate that required API keys are provided for the selected models.
    """
    errors = []
    
    # Check text model requirements
    if model_name_t in ['4o', '4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'o1', 'o3', 'o3-mini']:
        if not openai_api_key or not openai_api_key.strip():
            errors.append("OpenAI API key is required for GPT models")
    elif model_name_t in ['gemini', 'gemini-2.5-pro', 'gemini-2.5-flash']:
        if not gemini_api_key or not gemini_api_key.strip():
            errors.append("Gemini API key is required for Gemini models")
    elif model_name_t in ['qwen', 'qwen-plus', 'qwen-max', 'qwen-long']:
        if not qwen_api_key or not qwen_api_key.strip():
            errors.append("Qwen API key is required for Qwen models")
    elif model_name_t.startswith('openrouter_'):
        if not openrouter_api_key or not openrouter_api_key.strip():
            errors.append("OpenRouter API key is required for OpenRouter models")
    
    # Check vision model requirements
    if model_name_v in ['4o', '4o-mini']:
        if not openai_api_key or not openai_api_key.strip():
            errors.append("OpenAI API key is required for GPT vision models")
    elif model_name_v in ['gemini', 'gemini-2.5-pro', 'gemini-2.5-flash']:
        if not gemini_api_key or not gemini_api_key.strip():
            errors.append("Gemini API key is required for Gemini vision models")
    elif model_name_v in ['qwen-vl-max', 'qwen-2.5-vl-72b']:
        if not qwen_api_key or not qwen_api_key.strip():
            errors.append("Qwen API key is required for Qwen vision models")
    elif model_name_v.startswith('openrouter_'):
        if not openrouter_api_key or not openrouter_api_key.strip():
            errors.append("OpenRouter API key is required for OpenRouter vision models")
    
    return errors

# Global Variables
current_html_dir = None
preview_server = None
preview_port = None
template_preview_servers = []

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=current_html_dir, **kwargs)
    
    def log_message(self, format, *args):
        pass

def find_free_port(start_port=8000, max_attempts=100):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port")

def start_preview_server(html_dir):
    global current_html_dir, preview_server, preview_port
    stop_preview_server()
    current_html_dir = html_dir
    preview_port = find_free_port()
    preview_server = HTTPServer(('0.0.0.0', preview_port), CustomHTTPRequestHandler)
    server_thread = Thread(target=preview_server.serve_forever, daemon=True)
    server_thread.start()
    return preview_port

def stop_preview_server():
    global preview_server, preview_port
    if preview_server:
        preview_server.shutdown()
        preview_server = None
        preview_port = None

def start_ephemeral_server_for_dir(html_dir):
    port = find_free_port()
    class _TempHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=html_dir, **kwargs)
        def log_message(self, format, *args):
            pass
    srv = HTTPServer(('0.0.0.0', port), _TempHandler)
    t = Thread(target=srv.serve_forever, daemon=True)
    t.start()
    template_preview_servers.append((srv, port))
    return port

def stop_all_template_preview_servers():
    global template_preview_servers
    for srv, _ in template_preview_servers:
        try:
            srv.shutdown()
        except Exception:
            pass
    template_preview_servers = []

class GenerationArgs:
    def __init__(self, paper_path, model_name_t, model_name_v, template_root, 
                 template_dir, template_file, output_dir, style_preference, tmp_dir,
                 full_content_check_times, background_color, has_navigation, 
                 has_hero_section, title_color, page_density, image_layout, 
                 html_check_times, resume, human_input):
        self.paper_path = paper_path
        self.model_name_t = model_name_t
        self.model_name_v = model_name_v
        self.template_root = template_root
        self.template_dir = template_dir
        self.template_file = template_file
        self.output_dir = output_dir
        self.style_preference = style_preference
        self.tmp_dir = tmp_dir
        self.full_content_check_times = full_content_check_times
        self.background_color = background_color
        self.has_navigation = has_navigation
        self.has_hero_section = has_hero_section
        self.title_color = title_color
        self.page_density = page_density
        self.image_layout = image_layout
        self.html_check_times = html_check_times
        self.resume = resume
        self.human_input = human_input
        self.paper_name = None

# ==================== Formatting Functions ====================

def format_section_to_markdown(section_data):
    """
    Convert Section JSON to beautifully formatted Markdown
    
    Args:
        section_data: Section JSON data
    
    Returns:
        str: Formatted Markdown string
    """
    if not section_data:
        return "No data available"
    
    md_lines = []
    
    # Title
    md_lines.append("# 📄 Paper Page Structure Preview\n")
    
    # Basic Information
    if "title" in section_data:
        md_lines.append(f"## 📌 Title\n**{section_data['title']}**\n")
    
    if "authors" in section_data:
        md_lines.append(f"## 👥 Authors\n{section_data['authors']}\n")
    
    if "affiliation" in section_data:
        md_lines.append(f"## 🏛️ Affiliation\n{section_data['affiliation']}\n")
    
    # Other Sections
    md_lines.append("## 📑 Page Sections\n")
    
    section_count = 0
    for key, value in section_data.items():
        if key in ["title", "authors", "affiliation"]:
            continue
        
        section_count += 1
        
        # Section Title
        section_title = key.replace("_", " ").title()
        md_lines.append(f"### {section_count}. {section_title}\n")
        
        # Section Content
        if isinstance(value, dict):
            # If dictionary, process recursively
            for sub_key, sub_value in value.items():
                sub_title = sub_key.replace("_", " ").title()
                md_lines.append(f"**{sub_title}**: {sub_value}\n")
        elif isinstance(value, list):
            # If list
            for item in value:
                if isinstance(item, str):
                    md_lines.append(f"- {item}\n")
                elif isinstance(item, dict):
                    for k, v in item.items():
                        md_lines.append(f"- **{k}**: {v}\n")
        else:
            # Simple value
            md_lines.append(f"{value}\n")
        
        md_lines.append("")  # Empty line
    
    # Add Statistics
    md_lines.append("---\n")
    md_lines.append(f"**📊 Total {section_count} sections**\n")
    
    return "\n".join(md_lines)


def format_full_content_to_markdown(content_data, figures=None):
    """
    Convert Full Content JSON to beautifully formatted Markdown
    
    Args:
        content_data: Full Content JSON data
        figures: Images and tables data (optional)
    
    Returns:
        str: Formatted Markdown string
    """
    if not content_data:
        return "No data available"
    
    md_lines = []
    
    # Title
    md_lines.append("# 📄 Full Content Preview\n")
    
    # Basic Information
    if "title" in content_data:
        md_lines.append(f"# {content_data['title']}\n")
    
    if "authors" in content_data:
        md_lines.append(f"**Authors**: {content_data['authors']}\n")
    
    if "affiliation" in content_data:
        md_lines.append(f"**Affiliation**: {content_data['affiliation']}\n")
    
    md_lines.append("---\n")
    
    # Process Each Section
    section_count = 0
    image_count = 0
    table_count = 0
    
    for key, value in content_data.items():
        if key in ["title", "authors", "affiliation"]:
            continue
        
        section_count += 1
        
        # Section Title
        section_title = key.replace("_", " ").title()
        md_lines.append(f"## {section_count}. {section_title}\n")
        
        # Process Content
        if isinstance(value, dict):
            # Process dictionary type content
            for sub_key, sub_value in value.items():
                if sub_key.lower() in ['content', 'description', 'text']:
                    # Main text content
                    md_lines.append(f"{sub_value}\n")
                elif sub_key.lower() in ['image', 'figure', 'img']:
                    # Image
                    image_count += 1
                    if isinstance(sub_value, dict):
                        caption = sub_value.get('caption', f'Figure {image_count}')
                        path = sub_value.get('path', '')
                        md_lines.append(f"\n**🖼️ {caption}**\n")
                        if path:
                            md_lines.append(f"*Image path: `{path}`*\n")
                    else:
                        md_lines.append(f"\n**🖼️ Figure {image_count}**: {sub_value}\n")
                elif sub_key.lower() in ['table']:
                    # Table
                    table_count += 1
                    md_lines.append(f"\n**📊 Table {table_count}**\n")
                    if isinstance(sub_value, dict):
                        caption = sub_value.get('caption', f'Table {table_count}')
                        md_lines.append(f"*{caption}*\n")
                    else:
                        md_lines.append(f"{sub_value}\n")
                elif sub_key.lower() in ['code']:
                    # Code block
                    md_lines.append(f"\n```\n{sub_value}\n```\n")
                else:
                    # Other subtitles
                    sub_title = sub_key.replace("_", " ").title()
                    md_lines.append(f"\n### {sub_title}\n")
                    md_lines.append(f"{sub_value}\n")
        
        elif isinstance(value, list):
            # Process list type content
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    # Dictionary items in list
                    if 'title' in item or 'name' in item:
                        item_title = item.get('title', item.get('name', f'Item {idx+1}'))
                        md_lines.append(f"\n### {item_title}\n")
                    
                    for k, v in item.items():
                        if k not in ['title', 'name']:
                            if k.lower() in ['content', 'description', 'text']:
                                md_lines.append(f"{v}\n")
                            elif k.lower() in ['image', 'figure']:
                                image_count += 1
                                md_lines.append(f"\n**🖼️ Figure {image_count}**: {v}\n")
                            elif k.lower() == 'table':
                                table_count += 1
                                md_lines.append(f"\n**📊 Table {table_count}**: {v}\n")
                            else:
                                k_title = k.replace("_", " ").title()
                                md_lines.append(f"**{k_title}**: {v}\n")
                else:
                    # Simple list item
                    md_lines.append(f"- {item}\n")
        
        else:
            # Simple text value
            md_lines.append(f"{value}\n")
        
        md_lines.append("")  # Empty line between sections
    
    # Add Statistics
    md_lines.append("\n---\n")
    stats = []
    stats.append(f"📊 **Statistics**")
    stats.append(f"- Sections: {section_count}")
    if image_count > 0:
        stats.append(f"- Images: {image_count}")
    if table_count > 0:
        stats.append(f"- Tables: {table_count}")
    
    # If figures data is provided, add more information
    if figures:
        if 'images' in figures and figures['images']:
            stats.append(f"- Available images: {len(figures['images'])}")
        if 'tables' in figures and figures['tables']:
            stats.append(f"- Available tables: {len(figures['tables'])}")
    
    md_lines.append("\n".join(stats))
    md_lines.append("\n")
    
    return "\n".join(md_lines)

# ==================== Global State Management ====================

class GenerationState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.args = None
        self.paper_content = None
        self.figures = None
        self.generated_section = None
        self.text_page_content = None
        self.generated_content = None
        self.html_content = None
        self.html_file_path = None
        self.html_dir = None
        self.planner = None
        self.html_generator = None
        self.agent_config_t = None
        self.total_input_tokens_t = 0
        self.total_output_tokens_t = 0
        self.current_stage = "init"
        self.preview_url = None

state = GenerationState()

def create_project_zip(project_dir, output_dir, paper_name):
    """
    Create project archive
    
    Args:
        project_dir: Project directory path
        output_dir: Output directory
        paper_name: Paper name
    
    Returns:
        str: Archive path, None if failed
    """
    import zipfile
    
    zip_filename = f"{paper_name}_project_page.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    
    print(f"Creating project archive: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Traverse project directory, add all files
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate relative path
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
                    
        print(f"Archive created successfully: {zip_path}")
        
        # Get archive size
        zip_size = os.path.getsize(zip_path)
        zip_size_mb = zip_size / (1024 * 1024)
        print(f"Archive size: {zip_size_mb:.2f} MB")
        
        return zip_path
        
    except Exception as e:
        print(f"Archive creation failed: {e}")
        return None

def start_generation(pdf_file, model_name_t, model_name_v, template_root, 
                    template_dir, template_file, output_dir, style_preference, 
                    tmp_dir, full_content_check_times, background_color, 
                    has_navigation, has_hero_section, title_color, page_density, 
                    image_layout, html_check_times, resume, human_input, 
                    template_choice_value, openai_api_key, gemini_api_key, 
                    qwen_api_key, zhipuai_api_key, openrouter_api_key):
    """Start generation process"""
    if pdf_file is None:
        return "❌ Please upload a PDF file", gr.update(visible=False), "", "", gr.update(), gr.update(), ""
    
    # Validate API keys
    validation_errors = validate_api_keys(
        model_name_t, model_name_v, openai_api_key, gemini_api_key, 
        qwen_api_key, zhipuai_api_key, openrouter_api_key
    )
    
    if validation_errors:
        error_msg = "❌ API Key Validation Failed:\n" + "\n".join(f"• {error}" for error in validation_errors)
        return error_msg, gr.update(visible=False), "", "", gr.update(), gr.update(), ""
    
    state.reset()
    
    # Handle template selection
    if not (template_dir and str(template_dir).strip()):
        if not template_choice_value:
            stop_all_template_preview_servers()
            template_requirement = {
                "background_color": background_color,
                "has_hero_section": has_hero_section,
                "Page density": page_density,
                "image_layout": image_layout,
                "has_navigation": has_navigation,
                "title_color": title_color
            }
            try:
                matched = matching(template_requirement)
            except Exception as e:
                return f"❌ Template recommendation failed: {e}", gr.update(visible=False), "", "", gr.update(choices=[], value=None), gr.update(visible=False, value=""), ""
            
            html_finder_ = HtmlFinder()
            with open('templates/template_link.json','r') as f:
                template_link = json.load(f)
            previews = []
            for name in matched:
                t_dir = os.path.join(template_root, name)
                try:
                    html_path = html_finder_.find_html(t_dir)
                    if not os.path.exists(html_path):
                        continue
                    html_dir = os.path.dirname(os.path.abspath(html_path))
                    filename = os.path.basename(html_path)
                    port = start_ephemeral_server_for_dir(html_dir)
                    url = template_link[name]
                    previews.append((name, html_path, url))
                except Exception:
                    continue
            
            if not previews:
                return "❌ No previewable templates found", gr.update(visible=False), "", "", gr.update(choices=[], value=None), gr.update(visible=False, value=""), ""
            
            md_lines = ["### 🔍 Please select a template to preview before clicking **Start Generation**", ""]
            for name, _, url in previews:
                md_lines.append(f"- **{name}** → [{url}]({url})")
            md = "\n".join(md_lines)
            
            return "Recommended 3 templates, please select one to continue", gr.update(visible=False), "", "", gr.update(choices=[n for n, _, _ in previews], value=None), gr.update(visible=True, value=md), ""
        
        template_dir = os.path.join(template_root, template_choice_value)
    
    # Create arguments object
    args = GenerationArgs(
        paper_path=pdf_file.name,
        model_name_t=model_name_t,
        model_name_v=model_name_v,
        template_root=template_root,
        template_dir=template_dir,
        template_file=template_file,
        output_dir=output_dir,
        style_preference=style_preference,
        tmp_dir=tmp_dir,
        full_content_check_times=full_content_check_times,
        background_color=background_color,
        has_navigation=has_navigation,
        has_hero_section=has_hero_section,
        title_color=title_color,
        page_density=page_density,
        image_layout=image_layout,
        html_check_times=html_check_times,
        resume=resume,
        human_input=human_input
    )
    
    if not args.template_dir:
        return "❌ Please select a template", gr.update(visible=False), "", "", gr.update(), gr.update(), ""
    
    if not args.template_file:
        html_finder_ = HtmlFinder()
        args.template_file = html_finder_.find_html(args.template_dir)
    
    paper_name = args.paper_path.split('/')[-1].replace('.pdf', '') if '/' in args.paper_path else args.paper_path.replace('.pdf', '')
    args.paper_name = paper_name
    
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    try:
        # Initialization
        agent_config_t = get_agent_config_with_keys(
            args.model_name_t, openai_api_key, gemini_api_key, 
            qwen_api_key, zhipuai_api_key, openrouter_api_key
        )
        state.agent_config_t = agent_config_t
        state.args = args
        
        # Step 1: Parse PDF
        print("="*50)
        print("STEP 1: Parsing Research Paper")
        print("="*50)
        
        raw_content_path = f'project_contents/{args.paper_name}_raw_content.json'
        if not os.path.exists(raw_content_path):
            agent_config_v = get_agent_config_with_keys(
                args.model_name_v, openai_api_key, gemini_api_key, 
                qwen_api_key, zhipuai_api_key, openrouter_api_key
            )
            input_token, output_token, raw_result, images, tables = parse_paper_for_project_page(args, agent_config_t)
            state.total_input_tokens_t += input_token
            state.total_output_tokens_t += output_token
            raw_content_path, _ = save_parsed_content(args, raw_result, images, tables, input_token, output_token)
        
        with open(raw_content_path, 'r') as f:
            paper_content = json.load(f)
        
        images = paper_content.get('images', [])
        tables = paper_content.get('tables', [])
        figures = {'images': images, 'tables': tables}
        paper_content = paper_content.get('markdown_content', "")
        
        state.paper_content = paper_content
        state.figures = figures
        
        # Step 2: Filter content
        print("="*50)
        print("STEP 2: Filtering Content")
        print("="*50)
        
        planner = ProjectPageContentPlanner(agent_config_t, args)
        state.planner = planner
        
        paper_content, figures, input_token, output_token = planner.filter_raw_content(paper_content, figures)
        state.total_input_tokens_t += input_token
        state.total_output_tokens_t += output_token
        state.paper_content = paper_content
        state.figures = figures
        
        # Step 3: Generate Section
        print("="*50)
        print("STEP 3: Generating Sections")
        print("="*50)
        
        state.current_stage = "section"
        
        generated_section, input_token, output_token = generate_section_initial()
        state.total_input_tokens_t += input_token
        state.total_output_tokens_t += output_token
        
        # Use Markdown formatting
        section_display_md = format_section_to_markdown(generated_section)
        section_display_json = json.dumps(generated_section, indent=2, ensure_ascii=False)
        
        return (
            f"✅ Section generation completed, please review and provide feedback\n\nTokens: {input_token} → {output_token}",
            gr.update(visible=True),  # feedback_section
            section_display_md,       # Markdown format
            section_display_json,     # JSON format (hidden)
            gr.update(),
            gr.update(visible=False, value=""),
            ""
        )
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Generation failed: {str(e)}\n{traceback.format_exc()}"
        return error_msg, gr.update(visible=False), "", "", gr.update(), gr.update(), ""

def create_dynamic_page_dict(sections):
        poster_dict = {
            "title": "Title of the paper",
            "authors": "Authors of the paper",
            "affiliation": "Affiliation of the authors",
        }
        poster_dict.update(sections)
        return poster_dict
def generate_section_initial():
    """Generate initial Section"""
    import yaml
    from jinja2 import Environment, StrictUndefined
    from utils.wei_utils import account_token
    from utils.src.utils import get_json_from_response
    
    with open('utils/prompt_templates/page_templates/section_generation.yaml', 'r') as f:
        planner_config = yaml.safe_load(f)
    
    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(planner_config["template"])
    
    jinja_args = {
        'paper_content': state.paper_content,
        'json_format_example': json.dumps(state.paper_content, indent=2)
    }
    
    prompt = template.render(**jinja_args)
    
    state.planner.planner_agent.reset()
    response = state.planner.planner_agent.step(prompt)
    input_token, output_token = account_token(response)
    generated_section = get_json_from_response(response.msgs[0].content)
    generated_section = create_dynamic_page_dict(generated_section)
    state.generated_section = generated_section
    
    generated_path = f'project_contents/{state.args.paper_name}_generated_section.json'
    with open(generated_path, 'w') as f:
        json.dump(generated_section, f, indent=4)
    
    return generated_section, input_token, output_token

def submit_section_feedback(feedback_text):
    """Submit Section feedback"""
    if not feedback_text or feedback_text.strip().lower() == 'yes':
        # User satisfied, proceed to next stage
        result = proceed_to_text_content()
        status, fc_section_visible, fc_display_visible, fc_display_md, fc_display_json, fc_feedback_visible = result
        return (
            status,
            "",  # section_display_md clear
            "",  # section_display_json clear
            "",  # section_feedback_input clear
            gr.update(visible=False),  # feedback_section hide
            fc_section_visible,  # feedback_full_content show
            fc_display_visible,  # full_content_display_md show
            fc_display_md,  # full_content_display_md content
            fc_display_json,  # full_content_display_json content
            fc_feedback_visible  # full_content_feedback_input show
        )
    
    # User provides feedback, modify Section
    from camel.messages import BaseMessage
    from utils.wei_utils import account_token
    from utils.src.utils import get_json_from_response
    
    message = BaseMessage.make_assistant_message(
        role_name='User',
        content=f'human feedback: {feedback_text}\n\nPlease make modifications based on this feedback. Output format as specified above.'
    )
    response = state.planner.planner_agent.step(message)
    input_token, output_token = account_token(response)
    state.total_input_tokens_t += input_token
    state.total_output_tokens_t += output_token
    
    generated_section = get_json_from_response(response.msgs[0].content)
    generated_section = create_dynamic_page_dict(generated_section)
    state.generated_section = generated_section
    
    generated_path = f'project_contents/{state.args.paper_name}_generated_section.json'
    with open(generated_path, 'w') as f:
        json.dump(generated_section, f, indent=4)
    
    # Use Markdown formatting
    section_display_md = format_section_to_markdown(generated_section)
    section_display_json = json.dumps(generated_section, indent=2, ensure_ascii=False)
    
    return (
        f"✅ Section updated, please continue reviewing\n\nTokens: {input_token} → {output_token}",
        section_display_md,  # Markdown format
        section_display_json,  # JSON format
        "",  # Clear input box
        gr.update(visible=True),  # feedback_section keep visible
        gr.update(visible=False),  # feedback_full_content keep hidden
        gr.update(visible=False),  # full_content_display_md keep hidden
        "",  # full_content_display_md content
        "",  # full_content_display_json content
        gr.update(visible=False)  # full_content_feedback_input keep hidden
    )

def proceed_to_text_content():
    """Enter Text Content generation stage"""
    print("="*50)
    print("STEP 4: Generating Text Content")
    print("="*50)
    
    text_page_content, input_token, output_token = state.planner.text_content_generation(
        state.paper_content, state.figures, state.generated_section
    )
    state.total_input_tokens_t += input_token
    state.total_output_tokens_t += output_token
    state.text_page_content = text_page_content
    
    # Enter Full Content stage
    return proceed_to_full_content()

def proceed_to_full_content():
    """Enter Full Content generation stage"""
    print("="*50)
    print("STEP 5: Generating Full Content")
    print("="*50)
    
    state.current_stage = "full_content"
    
    generated_content, input_token, output_token = generate_full_content_initial()
    state.total_input_tokens_t += input_token
    state.total_output_tokens_t += output_token
    
    # Use Markdown formatting
    content_display_md = format_full_content_to_markdown(generated_content, state.figures)
    content_display_json = json.dumps(generated_content, indent=2, ensure_ascii=False)
    
    return (
        f"✅ Full Content generation completed, please review and provide feedback\n\nTokens: {input_token} → {output_token}",
        gr.update(visible=True),   # feedback_full_content show
        gr.update(visible=True),   # full_content_display_md show
        content_display_md,        # Markdown format
        content_display_json,      # JSON format
        gr.update(visible=True)    # full_content_feedback_input show
    )

def generate_full_content_initial():
    """Generate initial Full Content"""
    import yaml
    from jinja2 import Environment, StrictUndefined
    from utils.wei_utils import account_token
    from utils.src.utils import get_json_from_response
    
    with open('utils/prompt_templates/page_templates/full_content_generation.yaml', 'r') as f:
        planner_config = yaml.safe_load(f)
    
    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(planner_config["template"])
    
    jinja_args = {
        'paper_content': state.paper_content,
        'figures': json.dumps(state.figures, indent=2),
        'project_page_content': json.dumps(state.text_page_content, indent=2)
    }
    
    prompt = template.render(**jinja_args)
    
    state.planner.planner_agent.reset()
    response = state.planner.planner_agent.step(prompt)
    input_token, output_token = account_token(response)
    generated_content = get_json_from_response(response.msgs[0].content)
    
    state.generated_content = generated_content
    
    first_path = f'project_contents/{state.args.paper_name}_generated_full_content.v0.json'
    with open(first_path, 'w', encoding='utf-8') as f:
        json.dump(generated_content, f, ensure_ascii=False, indent=2)
    
    return generated_content, input_token, output_token

def submit_full_content_feedback(feedback_text):
    """Submit Full Content feedback"""
    if not feedback_text or feedback_text.strip().lower() == 'yes':
        # User satisfied, proceed to HTML generation
        result = proceed_to_html_generation()
        status, html_feedback_visible, preview_info, preview_url, open_btn_visible = result
        return (
            status,
            "",  # full_content_display_md clear
            "",  # full_content_display_json clear
            "",  # full_content_feedback_input clear
            gr.update(visible=False),  # feedback_full_content hide
            html_feedback_visible,  # feedback_html show
            preview_info,  # preview_info_display
            preview_url,  # preview_url_state
            open_btn_visible  # open_preview_btn show
        )
    
    # User provides feedback
    from camel.messages import BaseMessage
    from utils.wei_utils import account_token
    from utils.src.utils import get_json_from_response
    
    message = BaseMessage.make_assistant_message(
        role_name='User',
        content=f'human feedback: {feedback_text}\n\nPlease make modifications based on this feedback. Output format as specified above.'
    )
    response = state.planner.planner_agent.step(message)
    input_token, output_token = account_token(response)
    state.total_input_tokens_t += input_token
    state.total_output_tokens_t += output_token
    
    generated_content = get_json_from_response(response.msgs[0].content)
    state.generated_content = generated_content
    
    final_path = f'project_contents/{state.args.paper_name}_generated_full_content.json'
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(generated_content, f, ensure_ascii=False, indent=2)
    
    # Use Markdown formatting
    content_display_md = format_full_content_to_markdown(generated_content, state.figures)
    content_display_json = json.dumps(generated_content, indent=2, ensure_ascii=False)
    
    return (
        f"✅ Full Content updated, please continue reviewing\n\nTokens: {input_token} → {output_token}",
        content_display_md,  # Markdown format
        content_display_json,  # JSON format
        "",  # Clear input box
        gr.update(visible=True),  # feedback_full_content keep visible
        gr.update(visible=False),  # feedback_html keep hidden
        "",  # preview_info_display
        "",  # preview_url_state
        gr.update(visible=False)  # open_preview_btn keep hidden
    )

def proceed_to_html_generation():
    """Enter HTML generation stage"""
    print("="*50)
    print("STEP 6: Generating HTML")
    print("="*50)
    
    state.current_stage = "html"
    
    # Copy static files
    static_dir = copy_static_files(
        state.args.template_file, 
        state.args.template_dir, 
        state.args.output_dir, 
        state.args.paper_name
    )
    
    # Generate HTML
    html_relative_path = os.path.relpath(state.args.template_file, state.args.template_dir)
    html_dir = '/'.join(html_relative_path.strip().split('/')[:-1])
    state.html_dir = html_dir
    
    html_generator = ProjectPageHTMLGenerator(state.agent_config_t, state.args)
    state.html_generator = html_generator
    
    with open(state.args.template_file, 'r', encoding='utf-8') as file:
        html_template = file.read()
    
    # Create assets directory
    assets_dir = html_generator.create_assets_directory(state.args, html_dir, state.args.output_dir)
    
    # Generate HTML
    html_content, input_token, output_token = html_generator.generate_complete_html(
        state.args, state.generated_content, html_dir, html_template
    )
    state.total_input_tokens_t += input_token
    state.total_output_tokens_t += output_token
    
    # Save HTML (before table modification)
    html_dir_path = os.path.join(state.args.output_dir, state.args.paper_name, html_dir)
    os.makedirs(html_dir_path, exist_ok=True)
    
    html_file_path_no_modify = os.path.join(html_dir_path, 'index_no_modify_table.html')
    with open(html_file_path_no_modify, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    # Generate screenshot (before table modification)
    screenshot_path_no_modify = os.path.join(html_dir_path, 'page_final_no_modify_table.png')
    run_sync_screenshots(to_url(html_file_path_no_modify), screenshot_path_no_modify)
    
    # Modify tables
    html_content, input_token, output_token = html_generator.modify_html_table(html_content, html_dir)
    state.total_input_tokens_t += input_token
    state.total_output_tokens_t += output_token
    
    state.html_content = html_content
    
    # Save HTML (after table modification)
    html_file_path = os.path.join(html_dir_path, 'index.html')
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    state.html_file_path = html_file_path
    
    # Generate screenshot (after table modification)
    run_sync_screenshots(
        to_url(html_file_path), 
        os.path.join(html_dir_path, 'page_final.png')
    )
    
    # Start preview server
    html_full_dir = os.path.dirname(os.path.abspath(html_file_path))
    port = start_preview_server(html_full_dir)
    preview_url = f"http://localhost:{port}/index.html"
    state.preview_url = preview_url
    
    # Create preview info display
    preview_info = f"""
### 🌐 HTML Generation Completed

**Preview URL**: {preview_url}

**Instructions**:
1. Click the **"🌐 Open Preview in New Tab"** button below to view the generated webpage
2. Carefully review the page in the new tab
3. If satisfied, enter **'yes'** in the feedback box and submit
4. If modifications are needed, provide detailed feedback and submit

**Token Usage**: {input_token} → {output_token}
"""
    
    return (
        f"✅ HTML generation completed\n\nTokens: {input_token} → {output_token}",
        gr.update(visible=True),   # feedback_html show
        preview_info,              # preview_info_display
        preview_url,               # preview_url_state
        gr.update(visible=True)    # open_preview_btn show
    )

def submit_html_feedback(feedback_text):
    """Submit HTML feedback"""
    if not feedback_text or feedback_text.strip().lower() == 'yes':
        # User satisfied, complete generation
        result = finalize_generation()
        status, html_file = result
        return (
            status,
            "",  # preview_info_display clear
            "",  # html_feedback_input clear
            gr.update(visible=False),  # feedback_html hide
            gr.update(visible=False),  # open_preview_btn hide
            html_file  # html_file_output
        )
    
    # User provides feedback
    html_content, input_token, output_token = state.html_generator.modify_html_from_human_feedback(
        state.html_content, feedback_text
    )
    state.total_input_tokens_t += input_token
    state.total_output_tokens_t += output_token
    state.html_content = html_content
    
    # Save updated HTML
    html_dir_path = os.path.dirname(state.html_file_path)
    
    # Save as temporary version (for possible feedback iteration)
    import time
    timestamp = int(time.time())
    html_file_feedback = os.path.join(html_dir_path, f'index_feedback_{timestamp}.html')
    with open(html_file_feedback, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    # Also update main file
    with open(state.html_file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    # Regenerate screenshot
    screenshot_path = os.path.join(html_dir_path, 'page_final.png')
    try:
        run_sync_screenshots(to_url(state.html_file_path), screenshot_path)
    except Exception as e:
        print(f"Screenshot generation failed: {e}")
    
    # Update preview info
    preview_info = f"""
### 🌐 HTML Updated

**Preview URL**: {state.preview_url}

**Instructions**:
1. Click the **"🌐 Open Preview in New Tab"** button below to view the updated webpage
2. **Refresh the browser** to see the latest version
3. If satisfied, enter **'yes'** in the feedback box and submit
4. If further modifications are needed, continue providing feedback

**Token Usage**: {input_token} → {output_token}
"""
    
    return (
        f"✅ HTML updated, please refresh the preview page\n\nTokens: {input_token} → {output_token}",
        preview_info,              # preview_info_display
        "",  # Clear input box
        gr.update(visible=True),   # feedback_html keep visible
        gr.update(visible=True),   # open_preview_btn keep visible
        None  # html_file_output no download yet
    )

def finalize_generation():
    """Complete generation and save final results"""
    import time
    
    # Ensure final HTML is saved
    html_dir_path = os.path.dirname(state.html_file_path)
    
    # Save final version
    final_html_path = os.path.join(html_dir_path, 'index_final.html')
    with open(final_html_path, 'w', encoding='utf-8') as file:
        file.write(state.html_content)
    
    # Also update main file
    with open(state.html_file_path, 'w', encoding='utf-8') as file:
        file.write(state.html_content)
    
    # Save metadata
    metadata = state.html_generator.generate_metadata(state.generated_content, state.args)
    metadata_path = state.html_generator.save_metadata(metadata, state.args, state.args.output_dir)
    
    # Create README file
    readme_path = os.path.join(state.args.output_dir, state.args.paper_name, 'README.md')
    readme_content = f"""# {state.args.paper_name} - Project Page

## 📄 Project Information

- **Paper Name**: {state.args.paper_name}
- **Generation Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Text Model**: {state.args.model_name_t}
- **Vision Model**: {state.args.model_name_v}

## 🚀 Usage

1. Extract this archive to any directory
2. Open `index.html` to view the project page
3. All resources (CSS, images, etc.) are included

## 📁 File Structure

- `index.html` - Main page file
- `index_final.html` - Final confirmed version
- `assets/` - Image and table resources
- `css/` or `styles/` - Style files
- `js/` or `scripts/` - JavaScript files
- `metadata.json` - Page metadata
- `generation_log.json` - Generation log

## 💡 Tips

- Recommended browsers: Chrome, Firefox, Safari, Edge
- For web deployment, simply upload the entire folder
- Feel free to modify HTML and CSS for customization

---
Generated by Paper2ProjectPage
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Save generation log
    log_data = {
        'paper_name': state.args.paper_name,
        'paper_path': state.args.paper_path,
        'models': {
            'text_model': state.args.model_name_t,
            'vision_model': state.args.model_name_v
        },
        'token_usage': {
            'text_input_tokens': state.total_input_tokens_t,
            'text_output_tokens': state.total_output_tokens_t
        },
        'output_files': {
            'html_file': state.html_file_path,
            'final_html_file': final_html_path,
            'metadata_file': metadata_path,
            'readme_file': readme_path
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    log_path = f"{state.args.output_dir}/{state.args.paper_name}/generation_log.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)
    
    # Create project archive
    project_dir = os.path.join(state.args.output_dir, state.args.paper_name)
    zip_path = create_project_zip(project_dir, state.args.output_dir, state.args.paper_name)
    
    if zip_path and os.path.exists(zip_path):
        # Get archive size
        zip_size = os.path.getsize(zip_path)
        zip_size_mb = zip_size / (1024 * 1024)
        zip_filename = os.path.basename(zip_path)
        
        success_msg = f"""
✅ Project page generation completed!

📁 Output directory: {state.args.output_dir}/{state.args.paper_name}
🌐 HTML file: {state.html_file_path}
🌐 Final version: {final_html_path}
📋 Metadata: {metadata_path}
📖 README: {readme_path}
📊 Log file: {log_path}
📦 Archive: {zip_filename} ({zip_size_mb:.2f} MB)
🔢 Total token usage: {state.total_input_tokens_t} → {state.total_output_tokens_t}

🎉 All feedback completed, page successfully generated!
Click the button below to download the complete project archive (including HTML, CSS, images, README, and all resources).
"""
        
        return (
            success_msg,
            zip_path  # Return archive for download
        )
        
    else:
        error_msg = f"""
⚠️ Project page generated, but archive creation failed!

📁 Output directory: {state.args.output_dir}/{state.args.paper_name}
🌐 HTML file: {state.html_file_path}
📋 Metadata: {metadata_path}

You can manually retrieve all files from the output directory {project_dir}.
"""
        return (
            error_msg,
            state.html_file_path  # Return HTML file
        )

# ==================== Gradio Interface ====================

# Custom CSS for better English font rendering
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}

code, pre, .code {
    font-family: 'JetBrains Mono', 'Courier New', Consolas, Monaco, monospace !important;
}

h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

.markdown-text {
    line-height: 1.7 !important;
    font-size: 15px !important;
}

.gr-button {
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

.gr-input, .gr-textarea {
    font-size: 14px !important;
    line-height: 1.6 !important;
}

.gr-box {
    border-radius: 8px !important;
}

/* Better spacing for English content */
.gr-markdown p {
    margin-bottom: 0.8em !important;
}

.gr-markdown ul, .gr-markdown ol {
    margin-left: 1.2em !important;
}

.gr-markdown li {
    margin-bottom: 0.4em !important;
}
"""

with gr.Blocks(title="Paper2ProjectPage Generator", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    gr.Markdown("""
    # 📄 AutoPage Generator with Interactive Feedback
    
    Upload your research paper PDF and generate beautiful project pages through multi-round interactive feedback
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # PDF Upload
            pdf_input = gr.File(
                label="📎 Upload PDF Paper",
                file_types=[".pdf"],
                type="filepath"
            )
            
            gr.Markdown("### 🔑 API Keys Configuration")
            gr.Markdown("""
            **⚠️ Security Notice**: Your API keys are only stored in memory during the session and are never saved to disk.
            
            **💡 Note**: You only need to fill in ONE API key corresponding to the models you selected below.
            
            **📋 How to get API keys:**
            - **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
            - **Gemini**: Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
            - **Qwen**: Get your API key from [DashScope](https://dashscope.console.aliyun.com/apiKey)
            - **ZhipuAI**: Get your API key from [ZhipuAI Console](https://open.bigmodel.cn/usercenter/apikeys)
            - **OpenRouter**: Get your API key from [OpenRouter](https://openrouter.ai/keys)
            
            **🚀 For HuggingFace Spaces**: You can also set these as environment variables in your Space settings.
            """)
            
            with gr.Row():
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    type="password",
                    placeholder="sk-...",
                    info="Required for GPT models"
                )
                gemini_api_key = gr.Textbox(
                    label="Gemini API Key", 
                    value=os.getenv("GEMINI_API_KEY", ""),
                    type="password",
                    placeholder="AI...",
                    info="Required for Gemini models"
                )
            
            with gr.Row():
                qwen_api_key = gr.Textbox(
                    label="Qwen API Key",
                    value=os.getenv("QWEN_API_KEY", ""),
                    type="password", 
                    placeholder="sk-...",
                    info="Required for Qwen models"
                )
                zhipuai_api_key = gr.Textbox(
                    label="ZhipuAI API Key",
                    value=os.getenv("ZHIPUAI_API_KEY", ""),
                    type="password",
                    placeholder="...",
                    info="Required for GLM models"
                )
            
            openrouter_api_key = gr.Textbox(
                label="OpenRouter API Key",
                value=os.getenv("OPENROUTER_API_KEY", ""),
                type="password",
                placeholder="sk-or-...",
                info="Required for OpenRouter models"
            )
            
            gr.Markdown("### 🤖 Model Configuration")
            
            # Text Model Options
            text_model_options = [
                ("GPT-4o", "4o"),
                ("GPT-4o Mini", "4o-mini"),
                ("GPT-4.1", "gpt-4.1"),
                ("GPT-4.1 Mini", "gpt-4.1-mini"),
                ("O1", "o1"),
                ("O3", "o3"),
                ("O3 Mini", "o3-mini"),
                ("Gemini 2.5 Pro", "gemini"),
                ("Gemini 2.5 Pro (Alt)", "gemini-2.5-pro"),
                ("Gemini 2.5 Flash", "gemini-2.5-flash"),
                ("Qwen", "qwen"),
                ("Qwen Plus", "qwen-plus"),
                ("Qwen Max", "qwen-max"),
                ("Qwen Long", "qwen-long"),
                ("OpenRouter Qwen Plus", "openrouter_qwen-plus"),
                ("OpenRouter GPT-4o Mini", "openrouter_gpt-4o-mini"),
                ("OpenRouter Gemini 2.5 Flash", "openrouter_gemini-2.5-flash"),
                ("OpenRouter O3", "openrouter_openai/o3"),
                ("OpenRouter Claude Sonnet 4.5", "openrouter_claude-sonnet-4.5"),
            ]
            
            # Vision Model Options
            vision_model_options = [
                ("GPT-4o", "4o"),
                ("GPT-4o Mini", "4o-mini"),
                ("Gemini 2.5 Pro", "gemini"),
                ("Gemini 2.5 Pro (Alt)", "gemini-2.5-pro"),
                ("Gemini 2.5 Flash", "gemini-2.5-flash"),
                ("Qwen VL Max", "qwen-vl-max"),
                ("Qwen 2.5 VL 72B", "qwen-2.5-vl-72b"),
                ("OpenRouter Qwen VL 72B", "openrouter_qwen_vl_72b"),
                ("OpenRouter Qwen VL 7B", "openrouter_qwen_vl_7b"),
                ("OpenRouter Qwen VL Max", "openrouter_qwen-vl-max"),
                ("OpenRouter Gemini 2.5 Flash", "openrouter_gemini-2.5-flash"),
            ]
            
            with gr.Row():
                model_name_t = gr.Dropdown(
                    label="Text Model",
                    choices=text_model_options,
                    value="gemini",
                    info="Select model for text processing"
                )
                model_name_v = gr.Dropdown(
                    label="Vision Model", 
                    choices=vision_model_options,
                    value="gemini",
                    info="Select model for vision processing"
                )
            
            # Hidden backend configuration fields
            template_root = gr.Textbox(
                label="Template Root",
                value="templates",
                visible=False
            )
            template_dir = gr.Textbox(
                label="Template Directory",
                value="",
                visible=False
            )
            template_file = gr.Textbox(
                label="Template File",
                value="",
                visible=False
            )
            output_dir = gr.Textbox(
                label="Output Directory",
                value="generated_project_pages",
                visible=False
            )
            style_preference = gr.Textbox(
                label="Style Preference JSON",
                value="",
                visible=False
            )
            tmp_dir = gr.Textbox(
                label="Temporary Directory",
                value="tmp",
                visible=False
            )
            
            # Hidden parameters with default values
            resume = gr.Radio(
                label="Resume From Step",
                choices=['parse_pdf', 'generate_content','full_content_check', 'generate_html', 'html_check','modify_table','html_feedback'],
                value='parse_pdf',
                visible=False
            )
            
            human_input = gr.Radio(
                label="Enable Human Feedback",
                choices=[0, 1],
                value=1,
                visible=False
            )
            
            full_content_check_times = gr.Number(
                label="Full Content Check Times",
                value=1,
                precision=0,
                visible=False
            )
            
            html_check_times = gr.Number(
                label="HTML Check Times",
                value=1,
                precision=0,
                visible=False
            )
            
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 💡 User Guide
            
            1. **Upload PDF**: Select your research paper PDF file
            2. **Configure API Key**: Fill in ONE API key for your selected models
            3. **Select Models**: Choose text and vision models from dropdowns
            4. **Configure Style**: Adjust style preferences below
            5. **Start Generation**: Click the "Start Generation" button
            6. **Three-Stage Feedback**:
               - 📝 **Section Feedback**: Review the generated page structure (Markdown preview + JSON data)
               - 📄 **Full Content Feedback**: Review the generated complete content (Markdown preview + JSON data)
               - 🌐 **HTML Feedback**: View the generated webpage in a new tab
            7. **Download Results**: Download the complete project archive after completion
            
            ⚠️ **Tips**: 
            - Each stage supports multiple rounds of feedback until you're satisfied
            - Enter 'yes' to proceed to the next stage
            - The final ZIP download includes the complete project folder with all resources
            """)
            
            gr.Markdown("### 🎨 Style Configuration")
            
            with gr.Row():
                with gr.Column():
                    background_color = gr.Radio(
                        label="Background Color",
                        choices=["light", "dark"],
                        value="light",
                        info="Background color theme"
                    )
                    
                    has_navigation = gr.Radio(
                        label="Has Navigation",
                        choices=["yes", "no"],
                        value="yes",
                        info="Include navigation bar"
                    )
                    
                    has_hero_section = gr.Radio(
                        label="Has Hero Section",
                        choices=["yes", "no"],
                        value="yes",
                        info="Include hero/header section"
                    )
                
                with gr.Column():
                    title_color = gr.Radio(
                        label="Title Color",
                        choices=["pure", "colorful"],
                        value="pure",
                        info="Title color style"
                    )
                    
                    page_density = gr.Radio(
                        label="Page Density",
                        choices=["spacious", "compact"],
                        value="spacious",
                        info="Page spacing density"
                    )
                    
                    image_layout = gr.Radio(
                        label="Image Layout",
                        choices=["rotation", "parallelism"],
                        value="parallelism",
                        info="Image layout style"
                    )
            
            gr.Markdown("### 📁 Template Selection")
            
            template_choice = gr.Radio(
                label="Recommended Templates",
                choices=[],
                value=None,
                info="Select from recommended templates"
            )
            
            template_preview_links = gr.Markdown(
                value="",
                visible=False
            )
    
    # Start Generation Button
    start_btn = gr.Button("🚀 Start Generation", variant="primary", size="lg")
    
    # Status Output
    status_output = gr.Textbox(
        label="📊 Generation Status",
        lines=5,
        interactive=False
    )
    
    # Section Feedback Area
    with gr.Group(visible=False) as feedback_section:
        gr.Markdown("### 📝 Section Generation Results")
        gr.Markdown("Please review the generated section structure. If satisfied, enter **'yes'**, otherwise provide modification feedback:")
        
        with gr.Tabs():
            with gr.Tab("📖 Preview (Markdown)"):
                section_display_md = gr.Markdown(
                    label="Section Preview",
                    value=""
                )
            with gr.Tab("📋 Raw Data (JSON)"):
                section_display_json = gr.Code(
                    label="Section JSON",
                    language="json",
                    value="",
                    lines=15
                )
        
        section_feedback_input = gr.TextArea(
            label="Your Feedback",
            placeholder="Enter 'yes' to continue, or provide modification feedback...",
            lines=3
        )
        section_submit_btn = gr.Button("Submit Feedback", variant="primary")
    
    # Full Content Feedback Area
    with gr.Group(visible=False) as feedback_full_content:
        gr.Markdown("### 📄 Full Content Generation Results")
        gr.Markdown("Please review the generated full content. If satisfied, enter **'yes'**, otherwise provide modification feedback:")
        
        with gr.Tabs():
            with gr.Tab("📖 Preview (Markdown)"):
                full_content_display_md = gr.Markdown(
                    label="Full Content Preview",
                    value=""
                )
            with gr.Tab("📋 Raw Data (JSON)"):
                full_content_display_json = gr.Code(
                    label="Full Content JSON",
                    language="json",
                    value="",
                    lines=15
                )
        
        full_content_feedback_input = gr.TextArea(
            label="Your Feedback",
            placeholder="Enter 'yes' to continue, or provide modification feedback...",
            lines=3
        )
        full_content_submit_btn = gr.Button("Submit Feedback", variant="primary")
    
    # HTML Feedback Area
    with gr.Group(visible=False) as feedback_html:
        gr.Markdown("### 🌐 HTML Generation Results")
        
        # Preview Info Display
        preview_info_display = gr.Markdown(
            value="",
            label="Preview Information"
        )
        
        # Preview URL (hidden state for JS)
        preview_url_state = gr.Textbox(visible=False)
        
        # Open Preview in New Tab Button
        open_preview_btn = gr.Button(
            "🌐 Open Preview in New Tab",
            variant="secondary",
            size="lg",
            visible=False
        )
        
        gr.Markdown("---")
        
        # Feedback Input Area
        html_feedback_input = gr.TextArea(
            label="Your Feedback",
            placeholder="Enter 'yes' to finalize, or provide modification feedback...",
            lines=3
        )
        html_submit_btn = gr.Button("Submit Feedback", variant="primary")
    
    # Final Output
    html_file_output = gr.File(
        label="📥 Download Project Archive",
        interactive=False
    )
    
    # Bind Events
    start_btn.click(
        fn=start_generation,
        inputs=[
            pdf_input, model_name_t, model_name_v, template_root,
            template_dir, template_file, output_dir, style_preference,
            tmp_dir, full_content_check_times, background_color,
            has_navigation, has_hero_section, title_color, page_density,
            image_layout, html_check_times, resume, human_input,
            template_choice, openai_api_key, gemini_api_key, 
            qwen_api_key, zhipuai_api_key, openrouter_api_key
        ],
        outputs=[
            status_output,
            feedback_section,
            section_display_md,
            section_display_json,
            template_choice,
            template_preview_links,
            section_feedback_input
        ]
    )
    
    section_submit_btn.click(
        fn=submit_section_feedback,
        inputs=[section_feedback_input],
        outputs=[
            status_output,
            section_display_md,
            section_display_json,
            section_feedback_input,
            feedback_section,
            feedback_full_content,
            full_content_display_md,
            full_content_display_md,
            full_content_display_json,
            full_content_feedback_input
        ]
    )
    
    full_content_submit_btn.click(
        fn=submit_full_content_feedback,
        inputs=[full_content_feedback_input],
        outputs=[
            status_output,
            full_content_display_md,
            full_content_display_json,
            full_content_feedback_input,
            feedback_full_content,
            feedback_html,
            preview_info_display,
            preview_url_state,
            open_preview_btn
        ]
    )
    
    html_submit_btn.click(
        fn=submit_html_feedback,
        inputs=[html_feedback_input],
        outputs=[
            status_output,
            preview_info_display,
            html_feedback_input,
            feedback_html,
            open_preview_btn,
            html_file_output
        ]
    )
    
    # Open Preview Button - Use JavaScript to open in new tab
    open_preview_btn.click(
        fn=None,
        inputs=[preview_url_state],
        outputs=None,
        js="(url) => window.open(url, '_blank')"
    )

# Launch Application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )