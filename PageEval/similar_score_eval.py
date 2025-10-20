
import warnings
from transformers import logging

warnings.resetwarnings()
logging.set_verbosity_warning()

def calculate_similar_score(text1, text2,model_name='sentence-transformers/all-roberta-large-v1', device='cpu'):
    """
    Simple alternative using sentence-transformers if moverscore library is not available
    This is not true MoverScore but provides semantic similarity
    """
    warnings.filterwarnings("ignore")  
    logging.set_verbosity_error()     
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Load model
        model = SentenceTransformer(model_name_or_path =model_name, device=device,trust_remote_code=True)
        
        # Encode texts
        embeddings = model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return max(0,float(similarity))
        
    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None

def calculate_similar_score_between_json(json1,json2):
    
    json1 = {
        k: v for k,v in json1.items()
        if k.lower() not in {"title", "authors", "affiliation",'Title','Authors','Title and Authors'}
    }
    json2 = {
        k: v for k,v in json2.items()
        if k.lower() not in {"title", "authors", "affiliation",'Title','Authors','Title and Authors'}
    }
    result = {}
    for key in list(json1.keys()):
        if key in list(json2.keys()):
            result[key] = calculate_similar_score(json1[key],json2[key])
    
    return result
    
def calculate_similar_score_ours(json1,path):
    import json
    with open(path,'r') as f:
        json2 = json.load(f)
    return calculate_similar_score_between_json(json1,json2)


# Usage example for simple version
if __name__ == "__main__":
    text1 = '''
    The new AI system was designed to improve customer experience by providing faster and more accurate 
    responses. It analyzes user queries in real time and adapts its tone based on the detected emotion of the user.
    '''
    text2 = '''
    An advanced artificial intelligence platform has been developed to enhance user satisfaction by delivering quick 
    and precise answers. It processes questions instantly and adjusts its language according to the user’s emotional state.
    '''
    print(calculate_similar_score(text1,text2))

