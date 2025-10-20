from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


def get_ppl(
    text: str,
    model_name: str = "01-ai/Yi-6B",
    stride: int = 512,
) -> float:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto", 
    )
    model.eval()

   
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    max_len = 512
    if input_ids.size(0) <= max_len:

        with torch.no_grad():
            out = model(
                input_ids.unsqueeze(0).to(model.device), 
                labels=input_ids.unsqueeze(0).to(model.device)
            )
     
        return torch.exp(out.loss).item()


    nlls = []  
    for i in range(0, input_ids.size(0), stride):
       
        begin_loc = max(i + stride - max_len, 0)
        
        end_loc = min(i + stride, input_ids.size(0))
        
        trg_len = end_loc - i

      
        ids_chunk = input_ids[begin_loc:end_loc]
        labels = ids_chunk.clone()

       
        labels[:-trg_len] = -100  

        with torch.no_grad():
            out = model(
                ids_chunk.unsqueeze(0).to(model.device),
                labels=labels.unsqueeze(0).to(model.device)
            )
            nll = out.loss * trg_len  
        nlls.append(nll)
        if end_loc == input_ids.size(0):
            break


    ppl = torch.exp(torch.stack(nlls).sum() / input_ids.size(0))
    return ppl.item()

def get_full_content_ppl(path,model_name: str = "01-ai/Yi-6B",stride: int = 512,):
    import json
    with open(path,'r') as f:
        full_content = str(json.load(f))
    return get_ppl(full_content,model_name,stride)
        

# Usage example for simple version
if __name__=='__main__':
    
    text1 = '''
    The new AI system was designed to improve customer experience by providing faster and more accurate 
    responses. It analyzes user queries in real time and adapts its tone based on the detected emotion of the user.
    '''
    print(get_ppl(text1))


