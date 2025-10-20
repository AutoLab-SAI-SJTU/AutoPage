import re 
import json
import os
import math

def acc_qa(question_path: str,answer_path:str,compression = 2.71):
    """
    evla qa ACC
    """
    
    with open(question_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    detail_aspects = qa_data["detail"]["answers"]
    understanding_aspects = qa_data["understanding"]["answers"]
    detail_real_answers = [v[0] for k, v in detail_aspects.items()]

    understanding_real_answers = [v[0] for k, v in understanding_aspects.items()]

   
    with open(answer_path, 'r', encoding='utf-8') as f:
        pre_data = json.load(f)
    detail_pre_answers = [v['answer'] for k, v in pre_data["detail"].items()]

    understanding_pre_answers = [v['answer'] for k, v in pre_data["understanding"].items()]

    detail_acc = sum(1 for real, pre in zip(detail_real_answers, detail_pre_answers) if real == pre)/50
    understanding_acc = sum(1 for real, pre in zip(understanding_real_answers, understanding_pre_answers) if real == pre)/50
    lnC_detail_acc = detail_acc*math.log(compression)
    lnC_understanding_acc = understanding_acc*math.log(compression)
    return {
        'raw_acc':{
            'detail':detail_acc,
            'understanding':understanding_acc
        },
        'S_final':{
            'detail':lnC_detail_acc,
            'understanding':lnC_understanding_acc
        }
    }



# use example
if __name__=='__main__':
    
    question_path = 'qa_example.json'
    answer_path = 'answer.json'
    
    print(acc_qa(question_path,answer_path))

