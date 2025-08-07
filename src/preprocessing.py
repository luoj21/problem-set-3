'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import numpy as np

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Your code here
    model_pred_df = pd.read_csv("data/prediction_model_03.csv")
    genres_df = pd.read_csv("data/genres.csv")

    return model_pred_df, genres_df




def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Your code here

    # Initializing dictionaries for true counts, true positive, and true negative
    genres_list = genres_df['genre'].tolist()
    genre_true_counts = {key: None for key in genres_list}
    genre_tp_counts = {key: None for key in genres_list}
    genre_fp_counts = {key: None for key in genres_list}

    for key in genre_true_counts.keys():

        # For counting true counts
        true_count = 0 
        for idx, row in model_pred_df.iterrows():
            if key in row['actual genres']:
                true_count += 1
        genre_true_counts[key] = true_count

        # For counting TP counts
        tp_count = 0
        for idx, row in model_pred_df.iterrows():
            if key in row['predicted'] and row['actual genres'] and row['correct?'] == 1:
                tp_count += 1
        genre_tp_counts[key] = tp_count

        #For counting FP counts:
        fp_count = 0
        for idx, row in model_pred_df.iterrows():
            if key in row['predicted'] and key not in row['actual genres']:
                fp_count += 1
        genre_fp_counts[key] = fp_count

    
    return genres_list, genre_true_counts, genre_tp_counts, genre_fp_counts


if __name__ == "__main__":
    model_pred_df, genres_df = load_data()
    genres_list, genre_true_counts, genre_tp_counts, genre_fp_counts = process_data(model_pred_df, genres_df)
    print(genre_true_counts)
    print(genre_tp_counts)
    print(genre_fp_counts)


