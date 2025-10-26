# Simple domain detection - just 15 lines!
def detect_domain(content):
    """
    Simple keyword-based domain detection
    Returns: "academic" or "technical"
    """
    content_lower = content.lower()
    
    # Technical document keywords
    technical_words = ['api', 'endpoint', 'parameter', 'programming', 'installation', 
                      'configuration', 'troubleshooting', 'code example', 'getting started']
    
    # Academic paper keywords  
    academic_words = ['abstract', 'introduction', 'methodology', 'results', 'conclusion',
                     'references', 'literature review', 'hypothesis']
    
    tech_count = sum(1 for word in technical_words if word in content_lower)
    academic_count = sum(1 for word in academic_words if word in content_lower)
    
    return "technical" if tech_count > academic_count else "academic"