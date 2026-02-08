
import numpy as np
import pandas as pd
from math import log2

def rmse(true_ratings, pred_ratings):
    return np.sqrt(np.mean((np.array(true_ratings) - np.array(pred_ratings))**2))

def precision_at_k(recommended_list, relevent_items, k=10):
    recommended_k = recommended_list[:k]
    relevant_set = set(relevent_items)
    recommended_set = set(recommended_k)
    
    intersection = recommended_set.intersection(relevant_set)
    return len(intersection) / k

def recall_at_k(recommended_list, relevent_items, k=10):
    if not relevent_items:
        return 0.0
    recommended_k = recommended_list[:k]
    relevant_set = set(relevent_items)
    recommended_set = set(recommended_k)
    
    intersection = recommended_set.intersection(relevant_set)
    return len(intersection) / len(relevant_set)

def ndcg_at_k(recommended_list, relevent_items, k=10):
    recommended_k = recommended_list[:k]
    relevant_set = set(relevent_items)
    
    dcg = 0.0
    idcg = 0.0
    
    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            dcg += 1.0 / log2(i + 2)
            
    for i in range(min(len(relevant_set), k)):
        idcg += 1.0 / log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def coverage(recommended_lists, all_items):
    recommended_set = set()
    for l in recommended_lists:
        recommended_set.update(l)
    return len(recommended_set) / len(all_items)
