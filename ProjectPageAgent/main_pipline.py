"""
Main pipeline for Paper2ProjectPage.
Integrates all modules to generate project pages from research papers.
"""

import argparse
import json
import os
import time
from dotenv import load_dotenv
from pathlib import Path
import shutil
from ProjectPageAgent.parse_paper import parse_paper_for_project_page, save_parsed_content
from ProjectPageAgent.html_finder import HtmlFinder
from ProjectPageAgent.content_planner import ProjectPageContentPlanner
from ProjectPageAgent.html_generator import ProjectPageHTMLGenerator,to_url
from utils.wei_utils import get_agent_config
from ProjectPageAgent.content_planner import filter_references
from utils.src.utils import run_sync_screenshots

load_dotenv()

def matching(requirement):
    weight = {
        "background_color": 1.0,
        "has_hero_section": 0.75,
        "Page density": 0.85,
        "image_layout": 0.65,
        "title_color": 0.6,
        "has_navigation": 0.7
    }
    with open('tags.json', 'r') as f:
        template_tags = json.load(f)

    points = {}
    for name, tag in template_tags.items():
        for feature, value in tag.items():
            if requirement[feature] == value:
                if name not in points.keys():
                    points[name] = weight[feature]
                else:
                    points[name] += weight[feature]
    sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)
    return [template[0] for template in sorted_points[0:3]]

def copy_static_files(template_file_path, template_root_dir, output_dir, paper_name):
    
    print(f"Detecting Static files: {template_file_path}")
    os.makedirs(output_dir, exist_ok=True)
        
    # Create output directory for this specific project
    project_output_dir = f"{output_dir}/{paper_name}"
    os.makedirs(project_output_dir, exist_ok=True)
    
    # template_dir = os.path.dirname(template_file_path)
    static_dir = os.path.join(project_output_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)
    

    html_relative_path = os.path.relpath(template_file_path, template_root_dir)

    # template_static_dir = os.path.join(template_dir, 'static')
    if os.path.exists(template_root_dir) and os.path.isdir(template_root_dir):
        print(f"Found template dir: {template_root_dir}")
        try:
            shutil.copytree(template_root_dir, project_output_dir, dirs_exist_ok=True)
            os.remove(os.path.join(project_output_dir, html_relative_path))
            print(f"Copied template to: {project_output_dir}")
        except Exception as e:
            print(f"Failed to copy static files: {e}")

    try:
        with open(template_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"Failed to read template file: {e}")
        return
    
    return static_dir

def main():
    """Main pipeline for generating project pages from research papers."""
    parser = argparse.ArgumentParser(description='Paper2ProjectPage Generation Pipeline')
    parser.add_argument('--paper_path', type=str, required=True, help='Path to the research paper PDF')
    parser.add_argument('--model_name_t', type=str, default='4o', help='Text model name')
    parser.add_argument('--model_name_v', type=str, default='4o', help='Vision model name')
    parser.add_argument('--template_root', type=str, default="project_templates", help='Directory containing all templates')
    parser.add_argument('--template_dir', type=str, help='Directory of chosen template')
    parser.add_argument('--template_file', type=str, help='Path to a specific template file to use')
    parser.add_argument('--output_dir', type=str, default='generated_project_pages', help='Output directory for generated pages')
    parser.add_argument('--style_preference', type=str, default=None, help='Path to style preference JSON file')
    parser.add_argument('--tmp_dir', type=str, default='tmp', help='Temporary directory')
    parser.add_argument('--full_content_check_times', type=int, default='0', help='Temporary directory')
    parser.add_argument('--background_color', type=str, choices=['light', 'dark'], required=True,
                        help='Background color of generated project page')
    parser.add_argument('--has_navigation', type=str, choices=['yes', 'no'], required=True,
                        help='Is the generated project page has navigation')
    parser.add_argument('--has_hero_section', type=str, choices=['yes', 'no'], required=True,
                        help='Is the generated project page has hero section')
    parser.add_argument('--title_color', type=str, choices=['pure', 'colorful'], required=True,
                        help="Is the title's color of the project page is pure or colorful")
    parser.add_argument('--page_density', type=str, choices=['spacious', 'compact'], required=True,
                        help="The overall spacing tightness—amount of white space vs. information density")
    parser.add_argument('--image_layout', type=str, choices=['rotation', 'parallelism'], required=True,
                        help="The dominant arrangement style for images.")
    parser.add_argument('--html_check_times', type=int, default='1', help='Temporary directory')
    parser.add_argument(
        '--resume',
        type=str,
        choices=['parse_pdf', 'generate_content','full_content_check', 'generate_html', 'html_check','modify_table','html_feedback'],
        default='parse_pdf',
        help="From which step to resume: 'parse_pdf', 'generate_content','full_content_check', 'generate_html', 'html_check','modify_table','html_feedback'",
    )
    parser.add_argument('--human_input', type=str, default='1',choices=['0','1'] ,help='Human input for feedback')
    
    args = parser.parse_args()

    if not args.template_dir:
        template_requirement = {
            "background_color": args.background_color,
            "has_hero_section": args.has_hero_section,
            "Page density": args.page_density,
            "image_layout": args.image_layout,
            "has_navigation": args.has_navigation,
            "title_color": args.title_color
        }
        matched_template = matching(template_requirement)
        print('Below is names of the most matching 3 templates:')
        print('           '.join(matched_template))
        template_name = input('Please choose one from them, you can just input the name of your favorite template')
        while template_name not in matched_template:
            template_name = input('Please input the correct name of your favorite template!!')
        args.template_dir = os.path.join(args.template_root, template_name)

    # Extract html path from root path
    if not args.template_file:
        html_finder_ = HtmlFinder()
        args.template_file = html_finder_.find_html(args.template_dir)

    # Extract paper name from path
    paper_name = args.paper_path.split('/')[-1].replace('.pdf', '') if '/' in args.paper_path else args.paper_path.replace('.pdf', '')
    args.paper_name = paper_name
    
    print(f"Starting Paper2ProjectPage generation for: {paper_name}")
    print(f"Paper path: {args.paper_path}")
    print(f"Models: {args.model_name_t} (text), {args.model_name_v} (vision)")
    
    start_time = time.time()
    total_input_tokens_t = 0
    total_output_tokens_t = 0
    total_input_tokens_v = 0
    total_output_tokens_v = 0
    
    # Create temporary directory
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    try:
        # Get agent configurations
        agent_config_t = get_agent_config(args.model_name_t)
        agent_config_v = get_agent_config(args.model_name_v)
        
        # Step 1: Parse the research paper
        print("\n" + "="*50)
        print("STEP 1: Parsing Research Paper")
        print("="*50)

        raw_content_path = f'project_contents/{args.paper_name}_raw_content.json'
        if not os.path.exists(raw_content_path):
            print(f"Raw content does not exist at {raw_content_path}")


            input_token, output_token, raw_result, images, tables = parse_paper_for_project_page(args, agent_config_t)
            total_input_tokens_t += input_token
            total_output_tokens_t += output_token
            
            # Save parsed content
            raw_content_path, token_log_path = save_parsed_content(args, raw_result, images, tables, input_token, output_token)
            
            # Load parsed content
            with open(raw_content_path, 'r') as f:
                paper_content = json.load(f)
        else:
            print(f"Loading existing raw content from {raw_content_path}")
            with open(raw_content_path, 'r') as f:
                paper_content = json.load(f)
            # Load images and tables from the saved content
            images = paper_content.get('images', [])
            tables = paper_content.get('tables', [])
            token_log_path = raw_content_path.replace('_raw_content.json', '_parse_log.json')

        images = paper_content.get('images', [])
        tables = paper_content.get('tables', [])
        figures = {
            'images': images,
            'tables': tables
        }
        paper_content = paper_content.get('markdown_content', "")
    
        
        print("\n" + "="*50)
        print("STEP 2: Generate project page content")
        print("="*50)

        planner = ProjectPageContentPlanner(agent_config_t, args)
        figures_path = f'project_contents/{args.paper_name}_generated_filtered_figures.json'
        generated_section_path = f'project_contents/{args.paper_name}_generated_section.json'
        text_page_content_path = f'project_contents/{args.paper_name}_generated_text_content.json'
        generated_content_path = f'project_contents/{args.paper_name}_generated_full_content.json'
        if args.resume in ['parse_pdf','generate_content','full_content_check']:

            if args.resume != 'full_content_check':

                paper_content, figures, input_token, output_token = planner.filter_raw_content(paper_content, figures)
                total_input_tokens_t += input_token
                total_output_tokens_t += output_token
                
                generated_section, input_token, output_token = planner.section_generation(paper_content, figures)
                total_input_tokens_t += input_token
                total_output_tokens_t += output_token

                text_page_content, input_token, output_token = planner.text_content_generation(paper_content, figures, generated_section)
                total_input_tokens_t += input_token
                total_output_tokens_t += output_token

            else :
                print("Skipping content generation: filter_raw_content, section_generation, text_content_generation")
                print("Loading existing content from previous steps.")
                paper_content = filter_references(paper_content)
                with open(figures_path, 'r') as f:
                    figures = json.load(f)
                with open(generated_section_path, 'r') as f:
                    generated_section = json.load(f)        
                with open(text_page_content_path, 'r') as f:
                    text_page_content = json.load(f)

            generated_content, input_token, output_token = planner.full_content_generation(args, paper_content, figures, generated_section, text_page_content)
            total_input_tokens_t += input_token
            total_output_tokens_t += output_token

            print("\n" + "="*50)
            print("STEP 2.5: Copying Static Files")
            print("="*50)
            static_dir = copy_static_files(args.template_file, args.template_dir, args.output_dir, args.paper_name)
            
        else:
            print("Page content is already generated, loading existing content.")
            
            paper_content = filter_references(paper_content)
            with open(generated_section_path, 'r') as f:
                generated_section = json.load(f)        
            with open(text_page_content_path, 'r') as f:
                text_page_content = json.load(f)        
            with open(generated_content_path, 'r') as f:
                generated_content = json.load(f)

            static_dir = copy_static_files(args.template_file, args.template_dir, args.output_dir, args.paper_name)
            # static_dir = os.path.join(args.output_dir, args.paper_name, 'static')
        # Step 3: Generate HTML project page
        print("\n" + "="*50)
        print("STEP 3: Generating HTML Project Page")
        print("="*50)
        html_relative_path = os.path.relpath(args.template_file, args.template_dir)
        html_dir = '/'.join(html_relative_path.strip().split('/')[:-1])
        html_generator = ProjectPageHTMLGenerator(agent_config_t,args)
        with open(args.template_file, 'r', encoding='utf-8') as file:
                html_template = file.read()
        # Generate HTML
        if args.resume != 'modify_table' and args.resume != 'html_feedback':
            
            # Create assets directory and copy images
            assets_dir = html_generator.create_assets_directory(args, html_dir, args.output_dir)
            # Generate complete HTML
            html_content, input_token, output_token = html_generator.generate_complete_html(
                args, generated_content, html_dir, html_template
            )
            total_input_tokens_t += input_token
            total_output_tokens_t += output_token
            
            # Save HTML file
            html_file_path = os.path.join(args.output_dir, args.paper_name, html_dir, 'index_no_modify_table.html')
            with open(html_file_path,'w') as file:
                file.write(html_content)
            run_sync_screenshots(to_url(html_file_path), os.path.join(args.output_dir,args.paper_name, html_dir,'page_final_no_modify_table.png'))

        else:
            print(f'skip generate_html and html_check, load html from {os.path.join(args.output_dir, args.paper_name, html_dir, "index.html")}')
            assets_dir = os.path.join(args.output_dir, args.paper_name, html_dir,'assets')
            with open(os.path.join(args.output_dir,args.paper_name, html_dir,'index_no_modify_table.html'),'r') as file:
                html_content = file.read()
        
        if args.resume != 'html_feedback':
            html_content ,input_token,output_token = html_generator.modify_html_table(html_content,html_dir)
            total_input_tokens_t += input_token
            total_output_tokens_t += output_token
            html_file_path = os.path.join(args.output_dir, args.paper_name, html_dir, 'index_modify_table.html')
            with open(html_file_path,'w') as file:
                file.write(html_content)
            # html_file_path = html_generator.save_html_file(html_content, args, html_dir,args.output_dir)
        else:
            print("skipping modify_table,go to html_feedback")
            html_file_path = os.path.join(args.output_dir, args.paper_name, html_dir, 'index_modify_table.html')
            with open(html_file_path,'r') as file:
                html_content = file.read()

        print('-'*50)
        run_sync_screenshots(to_url(html_file_path), os.path.join(args.output_dir, args.paper_name, html_dir,'page_final.png'))
        if args.human_input == '1':
            human_feedback = input('Please view the final html in index.html,and image in page_final.png,If there are no problems, enter yes and press Enter.\n  If there are any problems, please give me feedback directly.\n')
            while human_feedback.lower() != 'yes':

                html_content ,input_token,output_token = html_generator.modify_html_from_human_feedback(html_content,human_feedback)
                total_input_tokens_t += input_token
                total_output_tokens_t += output_token
                with open(os.path.join(args.output_dir, args.paper_name, html_dir, 'index.html'),'w') as file:
                    file.write(html_content)
                run_sync_screenshots(to_url(os.path.join(args.output_dir, args.paper_name, html_dir, 'index.html')), os.path.join(args.output_dir, args.paper_name, html_dir,'page_final.png'))
                print('-'*50)
                human_feedback = input('Please view the final html in index.html,and image in page_final.png,If there are no problems, enter yes and press Enter. \n  If there are any problems, please give me feedback directly.\n')
            
        html_file_path = html_generator.save_html_file(html_content, args, html_dir,args.output_dir)

        # Generate and save metadata
        metadata = html_generator.generate_metadata(generated_content, args)
        metadata_path = html_generator.save_metadata(metadata, args, args.output_dir)
        
        # Step 4: Finalize and save logs
        print("\n" + "="*50)
        print("STEP 4: Finalizing Generation")
        print("="*50)
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Save generation log
        log_data = {
            'paper_name': paper_name,
            'paper_path': args.paper_path,
            'models': {
                'text_model': args.model_name_t,
                'vision_model': args.model_name_v
            },
            'token_usage': {
                'text_input_tokens': total_input_tokens_t,
                'text_output_tokens': total_output_tokens_t,
                'vision_input_tokens': total_input_tokens_v,
                'vision_output_tokens': total_output_tokens_v
            },
            'generation_time': time_taken,
            'output_files': {
                'html_file': html_file_path,
                'assets_dir': assets_dir,
                'static_dir': static_dir,
                'metadata_file': metadata_path
            },
            'content_files': {
                'raw_content': raw_content_path,
                'token_log': token_log_path
            }
        }
        
        log_path = f"{args.output_dir}/{args.paper_name}/generation_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        print(f"\n✅ Paper2ProjectPage generation completed successfully!")
        print(f"📁 Output directory: {args.output_dir}/{args.paper_name}")
        print(f"🌐 HTML file: {html_file_path}")
        print(f"📊 Assets directory: {assets_dir}")
        print(f"🎨 Static directory: {static_dir}")
        print(f"📋 Metadata file: {metadata_path}")
        print(f"⏱️  Total time: {time_taken:.2f} seconds")
        print(f"🔢 Token usage - Text: {total_input_tokens_t}→{total_output_tokens_t}, Vision: {total_input_tokens_v}→{total_output_tokens_v}")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
        raise

if __name__ == '__main__':
    main() 