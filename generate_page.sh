
dataset_dir="pdfs"
paper_name="AutoPage"
                    
python -m ProjectPageAgent.main_pipline\
    --paper_path="${dataset_dir}/${paper_name}.pdf" \
    --model_name_t="your_text_model" \
    --model_name_v="your_vlm_model" \
    --template_root="templates" \
    --template_dir="your_template_dir" \
    --template_file="your_template_file" \
    --output_dir="generated_project_pages" \
    --full_content_check_times=2 \
    --html_check_times=2 \
    --resume='parse_pdf' \
    --human_input='1' \
    --background_color='dark' \
    --has_navigation="yes" \
    --has_hero_section="no" \
    --title_color="colorful" \
    --page_density="compact" \
    --image_layout="rotation" 
    