import re
from collections import OrderedDict
from ProjectPageAgent.html_finder import HtmlFinder
import os



_LINK_CSS_RE = re.compile(
    r'''(?isx)
    <link[^>]*?                 
    href\s*=\s*                 
    (?:
        "([^"]+?\.css(?:\?[^"]*)?)" |
        '([^']+?\.css(?:\?[^']*)?)' |
        ([^\s"'=<>`]+?\.css(?:\?[^\s"'=<>`]*)?)
    )
    [^>]*?>
    '''
)


_IMPORT_CSS_RE = re.compile(
    r'''(?isx)
    @import
    \s+(?:url\()?
    \s*
    (?:
        "([^"]+?\.css(?:\?[^"]*)?)" |
        '([^']+?\.css(?:\?[^']*)?)' |
        ([^'")\s;]+?\.css(?:\?[^'")\s;]+)?)
    )        
    \s*
    \)?
    '''
)


def _first_nonempty(groups_list):
    out = []
    for groups in groups_list:
        for g in groups:
            if g:
                out.append(g)
                break
    return out

def extract_css_paths(html: str):

    links = _first_nonempty(_LINK_CSS_RE.findall(html))
    imports = _first_nonempty(_IMPORT_CSS_RE.findall(html))
    seen = OrderedDict()
    for u in links + imports:
        u = u.strip()
        if u and u not in seen:
            seen[u] = True
    return list(seen.keys())

def check_css(generated_html: str, template_html: str):
    generated_css = extract_css_paths(generated_html)
    template_css = extract_css_paths(template_html)
    print(f'num of css in generated page: {len(generated_css)}')
    print(f'num of css in template page: {len(template_css)}')
    template_css_name = {css.strip().split('/')[-1]: css for css in template_css}
    
    errors = {}
    for css in generated_css:
        if css.startswith('http'):
            continue
        if css not in template_css:
            match = template_css_name.get(css.strip().split('/')[-1], None)
            if match is not None:
                errors[css] = match
            else:
                print(f"[⚠️ Warning] Missing CSS match for {css}")
    
    new_html = generated_html
    for css, new_css in errors.items():
        if new_css:
            new_html = new_html.replace(css, new_css)
    
    return new_html





if __name__ == "__main__":

    templates_root = '/home/jimu/Project_resources/project_page/page_assets/'
    html_finder = HtmlFinder(specific_name='index.html')

    count = 0
    for page in os.listdir('generated_FastVGGT'):
        print(page)
        count += 1
        with open(html_finder.find_html(os.path.join('generated_FastVGGT', page)), 'r') as f:
            generated_html = f.read()

        with open(html_finder.find_html(os.path.join(templates_root, page)), 'r') as f:
            template_html = f.read()


        _ = check_css(generated_html, template_html, page)
    print(count)







