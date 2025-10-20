"""
Template analyzer for project page generation.
Analyzes existing project page templates to understand structure and style.
"""

import os
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path
import yaml
from jinja2 import Environment, StrictUndefined

class ProjectPageTemplateAnalyzer:
    """Analyzes project page templates to extract structure and styling patterns."""
    
    def __init__(self, template_dir="project_templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        self.templates = {}
        self.common_patterns = {}
        
    def analyze_html_template(self, html_file_path):
        """
        Analyze an HTML template file to extract structure and styling.
        
        Args:
            html_file_path: Path to the HTML template file
            
        Returns:
            dict: Analysis results including structure, styling, and patterns
        """
        try:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            analysis = {
                'file_path': html_file_path,
                'structure': self._extract_structure(soup),
                'styling': self._extract_styling(soup),
                'sections': self._extract_sections(soup),
                'components': self._extract_components(soup),
                'meta_info': self._extract_meta_info(soup)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing template {html_file_path}: {e}")
            return None
    
    def _extract_structure(self, soup):
        """Extract the overall structure of the HTML document."""
        structure = {
            'doctype': soup.find('!DOCTYPE') is not None,
            'html_lang': soup.html.get('lang', 'en') if soup.html else 'en',
            'head_sections': [],
            'body_sections': [],
            'main_content': None,
            'navigation': None,
            'footer': None
        }
        
        # Extract head sections
        if soup.head:
            for tag in soup.head.find_all(['meta', 'link', 'script', 'title']):
                structure['head_sections'].append({
                    'tag': tag.name,
                    'attrs': dict(tag.attrs)
                })
        
        # Extract body structure
        if soup.body:
            for section in soup.body.find_all(['header', 'nav', 'main', 'section', 'article', 'aside', 'footer']):
                structure['body_sections'].append({
                    'tag': section.name,
                    'id': section.get('id', ''),
                    'class': section.get('class', []),
                    'content_type': self._identify_content_type(section)
                })
        
        return structure
    
    def _extract_styling(self, soup):
        """Extract CSS styling information."""
        styling = {
            'inline_styles': [],
            'external_css': [],
            'color_scheme': [],
            'typography': {},
            'layout': {}
        }
        
        # Extract inline styles
        for tag in soup.find_all(style=True):
            styling['inline_styles'].append({
                'tag': tag.name,
                'style': tag.get('style', '')
            })
        
        # Extract external CSS links
        for link in soup.find_all('link', rel='stylesheet'):
            styling['external_css'].append(link.get('href', ''))
        
        # Extract color information
        color_pattern = re.compile(r'#[0-9a-fA-F]{3,6}|rgb\([^)]+\)|rgba\([^)]+\)')
        for tag in soup.find_all(style=True):
            colors = color_pattern.findall(tag.get('style', ''))
            styling['color_scheme'].extend(colors)
        
        # Extract typography patterns
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            font_size = re.search(r'font-size:\s*([^;]+)', tag.get('style', ''))
            if font_size:
                styling['typography'][tag.name] = font_size.group(1)
        
        return styling
    
    def _extract_sections(self, soup):
        """Extract content sections and their organization."""
        sections = []
        
        for section in soup.find_all(['section', 'article', 'div'], class_=True):
            section_info = {
                'tag': section.name,
                'id': section.get('id', ''),
                'classes': section.get('class', []),
                'content': self._extract_section_content(section),
                'images': self._extract_images(section),
                'tables': self._extract_tables(section)
            }
            sections.append(section_info)
        
        return sections
    
    def _extract_components(self, soup):
        """Extract reusable components and their patterns."""
        components = {
            'navigation': self._extract_navigation(soup),
            'hero_section': self._extract_hero_section(soup),
            'content_blocks': self._extract_content_blocks(soup),
            'image_galleries': self._extract_image_galleries(soup),
            'contact_forms': self._extract_contact_forms(soup)
        }
        
        return components
    
    def _extract_meta_info(self, soup):
        """Extract meta information and SEO elements."""
        meta_info = {
            'title': soup.title.string if soup.title else '',
            'meta_tags': [],
            'open_graph': {},
            'twitter_cards': {}
        }
        
        for meta in soup.find_all('meta'):
            meta_info['meta_tags'].append({
                'name': meta.get('name', ''),
                'content': meta.get('content', ''),
                'property': meta.get('property', '')
            })
            
            # Extract Open Graph tags
            if meta.get('property', '').startswith('og:'):
                meta_info['open_graph'][meta.get('property')] = meta.get('content', '')
            
            # Extract Twitter Card tags
            if meta.get('name', '').startswith('twitter:'):
                meta_info['twitter_cards'][meta.get('name')] = meta.get('content', '')
        
        return meta_info
    
    def _identify_content_type(self, element):
        """Identify the type of content in an element."""
        text = element.get_text().lower()
        
        if any(word in text for word in ['abstract', 'summary', 'overview']):
            return 'abstract'
        elif any(word in text for word in ['introduction', 'background']):
            return 'introduction'
        elif any(word in text for word in ['method', 'approach', 'methodology']):
            return 'methodology'
        elif any(word in text for word in ['result', 'experiment', 'evaluation']):
            return 'results'
        elif any(word in text for word in ['conclusion', 'discussion', 'future']):
            return 'conclusion'
        elif any(word in text for word in ['contact', 'author', 'team']):
            return 'contact'
        else:
            return 'general'
    
    def _extract_section_content(self, element):
        """Extract text content from a section."""
        content = {
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'code_blocks': []
        }
        
        for heading in element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            content['headings'].append({
                'level': int(heading.name[1]),
                'text': heading.get_text().strip()
            })
        
        for p in element.find_all('p'):
            content['paragraphs'].append(p.get_text().strip())
        
        for ul in element.find_all(['ul', 'ol']):
            items = [li.get_text().strip() for li in ul.find_all('li')]
            content['lists'].append({
                'type': ul.name,
                'items': items
            })
        
        for code in element.find_all(['code', 'pre']):
            content['code_blocks'].append({
                'type': code.name,
                'content': code.get_text().strip()
            })
        
        return content
    
    def _extract_images(self, element):
        """Extract image information from an element."""
        images = []
        for img in element.find_all('img'):
            images.append({
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
                'class': img.get('class', [])
            })
        return images
    
    def _extract_tables(self, element):
        """Extract table information from an element."""
        tables = []
        for table in element.find_all('table'):
            table_info = {
                'class': table.get('class', []),
                'headers': [],
                'rows': []
            }
            
            # Extract headers
            for th in table.find_all('th'):
                table_info['headers'].append(th.get_text().strip())
            
            # Extract rows
            for tr in table.find_all('tr'):
                row = [td.get_text().strip() for td in tr.find_all('td')]
                if row:
                    table_info['rows'].append(row)
            
            tables.append(table_info)
        
        return tables
    
    def _extract_navigation(self, soup):
        """Extract navigation structure."""
        nav = soup.find('nav')
        if nav:
            return {
                'links': [a.get('href', '') for a in nav.find_all('a')],
                'texts': [a.get_text().strip() for a in nav.find_all('a')],
                'structure': self._extract_nav_structure(nav)
            }
        return None
    
    def _extract_nav_structure(self, nav_element):
        """Extract the hierarchical structure of navigation."""
        structure = []
        for item in nav_element.find_all(['a', 'li'], recursive=False):
            if item.name == 'a':
                structure.append({
                    'type': 'link',
                    'text': item.get_text().strip(),
                    'href': item.get('href', '')
                })
            elif item.name == 'li':
                sub_items = []
                for sub_item in item.find_all('a'):
                    sub_items.append({
                        'text': sub_item.get_text().strip(),
                        'href': sub_item.get('href', '')
                    })
                structure.append({
                    'type': 'group',
                    'items': sub_items
                })
        return structure
    
    def _extract_hero_section(self, soup):
        """Extract hero section information."""
        hero = soup.find(['header', 'section'], class_=re.compile(r'hero|banner|intro'))
        if hero:
            return {
                'title': hero.find(['h1', 'h2']).get_text().strip() if hero.find(['h1', 'h2']) else '',
                'subtitle': hero.find(['h2', 'h3', 'p']).get_text().strip() if hero.find(['h2', 'h3', 'p']) else '',
                'background_image': hero.find('img').get('src', '') if hero.find('img') else '',
                'cta_buttons': [a.get_text().strip() for a in hero.find_all('a', class_=re.compile(r'btn|button'))]
            }
        return None
    
    def _extract_content_blocks(self, soup):
        """Extract content block patterns."""
        blocks = []
        for block in soup.find_all(['div', 'section'], class_=re.compile(r'content|block|section')):
            blocks.append({
                'classes': block.get('class', []),
                'content_type': self._identify_content_type(block),
                'has_images': bool(block.find('img')),
                'has_tables': bool(block.find('table')),
                'has_code': bool(block.find(['code', 'pre']))
            })
        return blocks
    
    def _extract_image_galleries(self, soup):
        """Extract image gallery patterns."""
        galleries = []
        for gallery in soup.find_all(['div', 'section'], class_=re.compile(r'gallery|carousel|slider')):
            images = gallery.find_all('img')
            galleries.append({
                'image_count': len(images),
                'layout': 'grid' if 'grid' in str(gallery.get('class', [])) else 'carousel',
                'images': [img.get('src', '') for img in images]
            })
        return galleries
    
    def _extract_contact_forms(self, soup):
        """Extract contact form patterns."""
        forms = []
        for form in soup.find_all('form'):
            form_info = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get'),
                'fields': []
            }
            
            for input_field in form.find_all(['input', 'textarea', 'select']):
                form_info['fields'].append({
                    'type': input_field.get('type', input_field.name),
                    'name': input_field.get('name', ''),
                    'placeholder': input_field.get('placeholder', ''),
                    'required': input_field.get('required') is not None
                })
            
            forms.append(form_info)
        
        return forms
    
    def analyze_multiple_templates(self, template_files):
        """
        Analyze multiple template files and find common patterns.
        
        Args:
            template_files: List of template file paths
            
        Returns:
            dict: Analysis results with common patterns
        """
        all_analyses = []
        
        for template_file in template_files:
            analysis = self.analyze_html_template(template_file)
            if analysis:
                all_analyses.append(analysis)
        
        # Find common patterns
        common_patterns = self._find_common_patterns(all_analyses)
        
        return {
            'individual_analyses': all_analyses,
            'common_patterns': common_patterns
        }
    
    def _find_common_patterns(self, analyses):
        """Find common patterns across multiple template analyses."""
        patterns = {
            'common_sections': [],
            'common_styles': [],
            'common_components': [],
            'color_schemes': [],
            'layout_patterns': []
        }
        
        # Analyze common sections
        all_sections = []
        for analysis in analyses:
            all_sections.extend(analysis['sections'])
        
        section_types = {}
        for section in all_sections:
            content_type = section.get('content_type', 'unknown')
            if content_type not in section_types:
                section_types[content_type] = 0
            section_types[content_type] += 1
        
        patterns['common_sections'] = [
            section_type for section_type, count in section_types.items()
            if count > len(analyses) * 0.5  # Appears in more than 50% of templates
        ]
        
        # Analyze common styles
        all_colors = []
        for analysis in analyses:
            all_colors.extend(analysis['styling']['color_scheme'])
        
        color_counts = {}
        for color in all_colors:
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1
        
        patterns['color_schemes'] = [
            color for color, count in color_counts.items()
            if count > len(analyses) * 0.3  # Appears in more than 30% of templates
        ]
        
        return patterns
    
    def save_analysis(self, analysis, output_path):
        """Save analysis results to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False 