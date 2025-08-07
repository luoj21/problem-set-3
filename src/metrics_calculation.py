'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Your code here

    # For micro:
    tp = sum(genre_tp_counts.values())
    fp  = sum(genre_fp_counts.values())
    fn =  sum(genre_true_counts.values()) - tp

    micro_precision = tp / (tp + fp)
    micro_recall = tp / (tp + fn)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    # For macro:
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []

    for genre in genre_list:
        tp = genre_tp_counts.get(genre)
        fp = genre_fp_counts.get(genre)
        fn = genre_true_counts.get(genre) - tp

        macro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        macro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

        macro_precision_list.append(macro_precision)
        macro_recall_list.append(macro_recall)
        macro_f1_list.append(macro_f1)


    return (micro_precision), (micro_recall), (micro_f1), macro_precision_list, macro_recall_list, macro_f1_list

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Your code here
    pred_rows = []
    true_rows = []

    for idx, row in model_pred_df.iterrows():
        true_row = [1 if genre in row['actual genres'] else 0 for genre in genre_list]
        pred_row = [1 if genre in row['predicted'] else 0 for genre in genre_list]
        true_rows.append(true_row)
        pred_rows.append(pred_row)

    # Create sparse matrix where the (i,j) entry is a 0 if sample i wasn't assigned class j, and 1 if it was
    true_matrix = pd.DataFrame(true_rows, columns=genre_list)
    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)

    # print(true_matrix)
    # print(pred_matrix)
    macro_prec, macro_rec, macro_f1, __ = precision_recall_fscore_support(true_matrix, pred_matrix, average='macro', zero_division=0.0)
    micro_prec, micro_rec, micro_f1, __ = precision_recall_fscore_support(true_matrix, pred_matrix, average='micro', zero_division=0.0)

    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1

