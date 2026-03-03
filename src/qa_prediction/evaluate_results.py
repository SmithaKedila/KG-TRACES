import json
import os
import re
import string

from tools import logger_factory

logger = logger_factory.setup_logger("evaluate_results")


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def contains_english_word(text):
    pattern = r'\b[a-zA-Z]+\b'
    return bool(re.search(pattern, text))

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0

    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision=matched/len(prediction)
    recall=matched/len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def evaluation_hit_k(predictions:list[str],answers:list[str],k:int):
  if k<=0:
     return 0
  topk_predictions=extract_topk_prediction(predictions,k)
  for a in answers:
    for p in topk_predictions:
       if match(p,a):
          return 1
  return 0

def eval_mrr(predictions:list[str],answers:list[str])->float:
    for i,p in enumerate(predictions):
       for a in answers:
         if match(p,a):
           return 1.0/(i+1)
    return 0.0 

def extract_answer(raw_prediction: str, skip_false_parse: bool=False) -> list[str]:
    if "**answer**" in raw_prediction.lower():
        predictions = raw_prediction.lower().split("**answer**")[1].strip("\n").strip(":").strip().split("\n")
    else:
        if skip_false_parse:
            return []
        else:
            predictions = raw_prediction.split("\n")
    return predictions

def eval_result(predict_file, cal_f1=True, topk=-1, is_tuned=False, skip_false_parse=False): #right now for checking the dual functionality
    if not predict_file.endswith('predictions.jsonl'):
        predict_file = os.path.join(predict_file, 'predictions.jsonl')
    eval_name = f"detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    acc_list, hit_list, f1_list, precision_list, recall_list,mrr_list = [], [], [], [], [],[]

    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except Exception as e:
                logger.info(f"Skipping line due to error: {e}")
                continue
            
            id = data['id']
            raw_prediction = data['prediction']
            if isinstance(raw_prediction, list) and len(raw_prediction) == 1:
                raw_prediction = raw_prediction[0]
            
            answer = list(set(data['ground_truth']))
            if isinstance(raw_prediction, list):
                full_prediction = raw_prediction
            else:
                full_prediction = extract_answer(raw_prediction, skip_false_parse) if is_tuned else raw_prediction.split("\n")

            if not full_prediction:
                continue
            if topk > 0:
                hit = evaluation_hit_k(full_prediction, answer, k=topk)
            else:
                prediction_str = ' '.join(full_prediction)
                hit = eval_hit(prediction_str, answer)

            f1_score, precision_score, recall_score = eval_f1(full_prediction, answer)
            prediction_str = ' '.join(full_prediction)
            acc = eval_acc(prediction_str, answer)
            mrr=eval_mrr(full_prediction,answer)
            acc_list.append(acc)
            hit_list.append(hit)
            f1_list.append(f1_score)
            precision_list.append(precision_score)
            recall_list.append(recall_score)
            mrr_list.append(mrr)
            f2.write(json.dumps({
                'id': id, 
                'prediction': full_prediction, 
                'ground_truth': answer, 
                'acc': acc, 
                'hit': hit, 
                'f1': f1_score, 
                'precision': precision_score, 
                'recall': recall_score,
                'MRR':mrr
            }) + '\n')

    if len(f1_list) > 0:
        result_str = (f"Accuracy: {sum(acc_list)*100/len(acc_list)} "
                      f"Hit: {sum(hit_list)*100/len(hit_list)} "
                      f"F1: {sum(f1_list)*100/len(f1_list)} "
                      f"Precision: {sum(precision_list)*100/len(precision_list)} "
                      f"Recall: {sum(recall_list)*100/len(recall_list)}"
                      f"MRR:{sum(mrr_list)*100/len(mrr_list):.2f}")
    else:
        result_str = f"Accuracy: {sum(acc_list)*100/len(acc_list)} Hit: {sum(hit_list)*100/len(hit_list)}"
    
    logger.info(result_str)
    
    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)
