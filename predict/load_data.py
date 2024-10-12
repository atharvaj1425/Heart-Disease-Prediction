# predict/load_data.py
import pandas as pd
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'heart.csv')

def load_dataset():
    return pd.read_csv(DATA_FILE)
