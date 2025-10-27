"""
Content planner for project page generation.
Plans the structure and content organization for the project page.
"""

import json
import yaml
import os
from jinja2 import Environment, StrictUndefined
from camel.models import ModelFactory
from camel.agents import ChatAgent
from utils.wei_utils import  account_token
from utils.src.utils import get_json_from_response
from camel.messages import BaseMessage
from rich import print
from rich.pretty import Pretty
import base64
from camel.messages import BaseMessage
from camel.models import ModelFactory
from .domain_detector import detect_domain

def filter_references(md_content: str) -> str:
  
    lines = md_content.splitlines()
    result_lines = []
    for line in lines:
        if line.strip().lower().startswith("## references"):
            break  
        result_lines.append(line)
    return "\n".join(result_lines)

class ProjectPageContentPlanner:
    """Plans the content structure and organization for project pages."""
    
    def __init__(self, agent_config, args):
        self.agent_config = agent_config
        self.args = args
        self.planner_agent = self._create_planner_agent()      
        self.reviewer_agent = self._create_reviewer_agent()   
        os.makedirs('project_contents', exist_ok=True)
        
    def _create_planner_agent(self):
        """Create the content planning (generation) agent."""
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

        
        system_message = """You are a helpful academic expert and web developer, who is specialized in generating a paper project page, from given research paper's contents and figures."""
        
        return ChatAgent(
            system_message=system_message,
            model=model,
            message_window_size=10,
            token_limit=self.agent_config.get('token_limit', None)
        )

    def _create_reviewer_agent(self):
       
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
        
        reviewer_system = (
            "You are a precise, constructive reviewer of generated project pages. "
        )
        return ChatAgent(
            system_message=reviewer_system,
            model=model,
            message_window_size=10,
            token_limit=self.agent_config.get('token_limit', None)
        )

    def _render_generation_prompt(self, paper_content, figures, text_page_content, template_str):
      
        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(template_str)
        jinja_args = {
            'paper_content': paper_content,
            'figures': json.dumps(figures, indent=2),
            'project_page_content': json.dumps(text_page_content, indent=2),
        }
        return template.render(**jinja_args)

    def _build_reviewer_prompt(self, paper_content, figures, text_page_content, generated_json):
       
        with open('utils/prompt_templates/page_templates/full_content_review.yaml', 'r') as f:
            planner_config = yaml.safe_load(f)
        
        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(planner_config["template"])
        
        jinja_args = {
            'paper_content': paper_content,
            'figures': json.dumps(figures['images'], indent=2),
            'tables': json.dumps(figures['tables'], indent=2),
            "generated_content": generated_json
        }
        
        prompt = template.render(**jinja_args)

        return prompt

    def _build_revision_prompt(self, review_json):
        with open('utils/prompt_templates/page_templates/full_content_revise.yaml', 'r') as f:
            planner_config = yaml.safe_load(f)
        
        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(planner_config["template"])
        
        jinja_args = {
            "review_content": json.dumps(review_json, indent=2)
        }
        
        prompt = template.render(**jinja_args)

        return prompt

    def _build_revision_prompt_with_resume(self, review_json, current_content, figures):
        with open('utils/prompt_templates/page_templates/full_content_revise_with_resume.yaml', 'r') as f:
            planner_config = yaml.safe_load(f)

        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(planner_config["template"])

        print(review_json)

        jinja_args = {
            "review_content": json.dumps(review_json, indent=2),
            "figures": json.dumps(figures, indent=2),
            "current_content": current_content
        }

        prompt = template.render(**jinja_args)

        return prompt

    def full_content_generation(
        self,
        args,
        paper_content,
        figures,
        generated_section,
        text_page_content,
    ):
        """
        Plan + Generate -> Review -> Revise 
        
        Args:
            paper_content: parsed paper content
            figures: list/dict of figures
            generated_section: format_instructions / schema hints
            text_page_content: initial text-only page structure
        
        Returns:
            tuple: (final_generated_content_json, input_token_total, output_token_total)
        """
        if args.resume in ['parse_pdf','generate_content']:

            print("full content generation start")

            with open('utils/prompt_templates/page_templates/full_content_generation.yaml', 'r') as f:
                planner_config = yaml.safe_load(f)
            
            jinja_env = Environment(undefined=StrictUndefined)
            template = jinja_env.from_string(planner_config["template"])
            
            jinja_args = {
                'paper_content': paper_content,
                'figures': json.dumps(figures, indent=2),
                'project_page_content': json.dumps(text_page_content, indent=2)
            }
            
            prompt = template.render(**jinja_args)
            
            self.planner_agent.reset()
            response = self.planner_agent.step(prompt)
            
            gen_in_tok, gen_out_tok = account_token(response)

            current_output = get_json_from_response(response.msgs[0].content)

            first_path = f'project_contents/{self.args.paper_name}_generated_full_content.v0.json'
            with open(first_path, 'w', encoding='utf-8') as f:
                json.dump(current_output, f, ensure_ascii=False, indent=2)
            print(f"  - Initial generation saved: {first_path}")

            total_in_tok, total_out_tok = gen_in_tok, gen_out_tok
        else:
            print("Skipping initial full content generation, loading existing content.")
            with open(f'project_contents/{self.args.paper_name}_generated_full_content.v0.json', 'r', encoding='utf-8') as f:
                current_output = json.load(f)
            total_in_tok, total_out_tok = 0, 0

        for it in range(0, args.full_content_check_times):
            # check
            self.reviewer_agent.reset()

            review_prompt = self._build_reviewer_prompt(
                paper_content=paper_content,
                figures=figures,
                text_page_content=text_page_content,
                generated_json=current_output
            )
            review_resp = self.reviewer_agent.step(review_prompt)
            rin, rout = account_token(review_resp)

            review_json = get_json_from_response(review_resp.msgs[0].content)

            review_path = f'project_contents/{self.args.paper_name}_review.iter{it}.json'
            with open(review_path, 'w', encoding='utf-8') as f:
                json.dump(review_json, f, ensure_ascii=False, indent=2)
            print(f"  - Review saved: {review_path}")

            total_in_tok += rin
            total_out_tok += rout

            if args.resume != 'full_content_check':
                revision_prompt = self._build_revision_prompt(
                    review_json=review_json
                )

            else:
                revision_prompt = self._build_revision_prompt_with_resume(
                    review_json=review_json,
                    current_content=current_output,
                    figures=figures
                )
            rev_resp = self.planner_agent.step(revision_prompt)
            rin2, rout2 = account_token(rev_resp)

            revised_output = get_json_from_response(rev_resp.msgs[0].content)

            out_path = f'project_contents/{self.args.paper_name}_generated_full_content.v{it+1}.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(revised_output, f, ensure_ascii=False, indent=2)
            print(f"  - Revised generation saved: {out_path}")

            total_in_tok += rin2
            total_out_tok += rout2
            current_output = revised_output
        if self.args.human_input == '1':
            print('-'*50)
            print(Pretty(current_output, expand_all=True))
            print('-'*50)
            user_feedback = input('The above is the final generated full content! If you are satisfied with the generated content, enter yes\n If not, enter your feedback.\n')
            while user_feedback.lower() != 'yes':
                message = BaseMessage.make_assistant_message(
                    role_name='User',
                    content='human feedback'+user_feedback +"The above is human feedback. Please make modifications based on this feedback and the original content.The output format is as specified above."
                )
                response = self.planner_agent.step(message)
                current_output = get_json_from_response(response.msgs[0].content)
                print('-'*50)
                print(Pretty(current_output, expand_all=True))
                print('-'*50)
                user_feedback = input('The above is the final generated full content! If you are satisfied with the generated content, enter yes. \n If not, enter your feedback.\n')
                in_tok, out_tok = account_token(response)
                total_in_tok += in_tok
                total_out_tok += out_tok
            
        # 4) 最终保存（保持你原有的命名）
        final_path = f'project_contents/{self.args.paper_name}_generated_full_content.json'
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(current_output, f, ensure_ascii=False, indent=2)
        print(f"full content generation completed. Tokens: {total_in_tok} -> {total_out_tok}")
        print(f"  - Final content: {final_path}")

        return current_output, total_in_tok, total_out_tok
    
    
    def section_generation(self, paper_content, figures):
        """
        Plan the content structure for the project page.
    
        Args:
            paper_content: Parsed paper content
        
            Returns:
        dict: project page content
        """
    
        # Detect document domain for template selection
        domain = detect_domain(paper_content)
        print(f"🎯 Detected domain: {domain}")
    
        # Choose template based on domain
        if domain == "technical":
            template_file = 'utils/prompt_templates/page_templates/adaptive_sections.yaml'
        else:
            template_file = 'utils/prompt_templates/page_templates/section_generation.yaml'
    
        # Load planning prompt template
        with open(template_file, 'r') as f:
            planner_config = yaml.safe_load(f)
    
        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(planner_config["template"])

        json_format_example = """
```json
{
    "Introduction": "Brief overview of the paper's main topic and objectives.",
    "Methodology": "Description of the methods used in the research.",
    "Results": "Summary of the key findings and results."
}
        
        # Prepare template arguments
        jinja_args = {
            'paper_content': paper_content,
            'json_format_example': json.dumps(paper_content, indent=2)
        }

        # Add domain to template args if using adaptive template
        if domain == "technical":
            jinja_args['domain'] = domain

        prompt = template.render(**jinja_args)

        # Generate content plan
        self.planner_agent.reset()
        response = self.planner_agent.step(prompt)
        input_token, output_token = account_token(response)
        generated_section = get_json_from_response(response.msgs[0].content)

        if self.args.human_input == '1':
            print('-'*50)
            print(Pretty(generated_section, expand_all=True))
            print('-'*50)
            user_feedback = input('The above is the generated section! If you are satisfied with the generated section, enter yes. \nIf not, enter your feedback.\n')
            while user_feedback.lower() != 'yes':
                message = BaseMessage.make_assistant_message(
                    role_name='User',
                    content='human feedback'+user_feedback +"The above is human feedback. Please make modifications based on this feedback and the original content.The output format is as specified above."
                )
                response = self.planner_agent.step(message)
                generated_section = get_json_from_response(response.msgs[0].content)
                print('-'*50)
                print(Pretty(generated_section, expand_all=True))
                print('-'*50)
                user_feedback = input('The above is the generated section! If you are satisfied with the generated section, enter yes. \nIf not, enter your feedback.\n')
                in_tok, out_tok = account_token(response)
                input_token += in_tok
                output_token += out_tok

        print(f"section planning completed. Tokens: {input_token} -> {output_token}")

        def create_dynamic_page_dict(sections: dict[str, str]) -> dict[str, str]:
            poster_dict = {
                "title": "Title of the paper",
                "authors": "Authors of the paper, Each author must be accompanied by the superscript number(s) of their corresponding affiliation(s).",
                "affiliation": "Affiliation of the authors, each affiliation must be accompanied by the corresponding superscript number.",
            }

            poster_dict.update(sections)
            return poster_dict

        generated_section = create_dynamic_page_dict(generated_section)

        # Save generated content
        generated_path = f'project_contents/{self.args.paper_name}_generated_section.json'
        with open(generated_path, 'w') as f:
            json.dump(generated_section, f, indent=4)

        print(f"  - Generated section plan: {generated_path}")

        return generated_section, input_token, output_token

    def text_content_generation(self, paper_content, figures, generated_section):
        """
        Plan the content structure for the project page.
        
        Args:
            paper_content: Parsed paper content
            
        Returns:
            dict: project page content
        """

        # Delete tags in figures
        figures_ = {}
        figures_['images'] = [{k: v for k, v in value.items() if k != 'tag'} for value in figures['images'].values()]
        figures_['tables'] = [{k: v for k, v in value.items() if k != 'tag'} for value in figures['tables'].values()]

        # Load planning prompt template
        with open('utils/prompt_templates/page_templates/text_content_generation.yaml', 'r') as f:
            planner_config = yaml.safe_load(f)
        
        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(planner_config["template"])
        
        # Prepare template arguments
        jinja_args = {
            'paper_content': paper_content,
            'figures': json.dumps(figures_, indent=2),
            'format_instructions': json.dumps(generated_section, indent=2)
        }
        
        prompt = template.render(**jinja_args)
        
        # Generate content plan
        self.planner_agent.reset()
        response = self.planner_agent.step(prompt)
        input_token, output_token = account_token(response)
        
        generated_text_content = get_json_from_response(response.msgs[0].content)
        
        print(f"text content generation completed. Tokens: {input_token} -> {output_token}")

        # Save generated content
        generated_path = f'project_contents/{self.args.paper_name}_generated_text_content.json'
        with open(generated_path, 'w') as f:
            json.dump(generated_text_content, f, indent=4)
        
        print(f"  - Generated text content: {generated_path}")
        
        return generated_text_content, input_token, output_token

    def filter_raw_content(self, paper_content, figures):
        paper_content = filter_references(paper_content)
        # Load planning prompt template
        with open('utils/prompt_templates/page_templates/filter_figures.yaml', 'r') as f:
            planner_config = yaml.safe_load(f)
        
        jinja_env = Environment(undefined=StrictUndefined)
        template = jinja_env.from_string(planner_config["template"])
        
        # Prepare template arguments
        jinja_args = {
            'paper_content': paper_content,
            'figures': json.dumps(figures, indent=2),
        }
        
        prompt = template.render(**jinja_args)
        
        # Generate filtered figures
        self.planner_agent.reset()
        response = self.planner_agent.step(prompt)
        input_token, output_token = account_token(response)
        filtered_figures = get_json_from_response(response.msgs[0].content)
        #print(filtered_figures)

        def remove_items_without_section(data: dict) -> dict:
            
            for key in ["images", "tables"]:
                if key in data and isinstance(data[key], dict):
                    data[key] = {
                        k: v for k, v in data[key].items()
                        if v.get("original_section") is not None
                    }
            return data

        filtered_figures = remove_items_without_section(filtered_figures)
        
        print(f"filtered figures generation completed. Tokens: {input_token} -> {output_token}")

        # Save generated filtered figures
        generated_path = f'project_contents/{self.args.paper_name}_generated_filtered_figures.json'
        with open(generated_path, 'w') as f:
            json.dump(filtered_figures, f, indent=4)
        
        print(f"  - Generated filtered figures: {generated_path}")
        
        return paper_content, filtered_figures, input_token, output_token


        