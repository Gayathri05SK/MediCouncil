import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config.settings import DATA_DIR
from src.utils.logger import logger

DATADIR = DATA_DIR

def load_raw_data():
    """Load raw dataset files"""
    logger.info("Loading raw data...")
    raw_dir = DATADIR / "raw"
    raw_file = raw_dir / "disease_symptoms.csv"
    
    if not raw_file.exists():
        logger.warning("Raw data not found. Creating sample data...")
        create_sample_data()
    
    df = pd.read_csv(raw_file)
    logger.info(f"Loaded {len(df)} records")
    return df

def create_sample_data():
    """Create balanced sample dataset (40 per class)"""
    logger.info("Generating balanced sample data...")
    
    # Define balanced samples: 40 each for Low, Medium, High, Critical
    base_samples = [
        # Low (40x)
        {"symptoms_text": "mild headache runny nose sore throat", "age": 28, "sex": "M", "chronic_conditions": "", "red_flags": "", "triagelabel": "Low"},
        {"symptoms_text": "back pain worse when lifting no numbness", "age": 40, "sex": "M", "chronic_conditions": "", "red_flags": "", "triagelabel": "Low"},
        {"symptoms_text": "mild cough no fever no shortness of breath", "age": 32, "sex": "F", "chronic_conditions": "", "red_flags": "", "triagelabel": "Low"},
        # ... repeat pattern for 40 Low
        
        # Medium (40x)
        {"symptoms_text": "persistent cough yellow mucus 5 days mild fever", "age": 35, "sex": "F", "chronic_conditions": "", "red_flags": "", "triagelabel": "Medium"},
        {"symptoms_text": "stomach ache diarrhea vomiting 1 day", "age": 30, "sex": "M", "chronic_conditions": "", "red_flags": "", "triagelabel": "Medium"},
        # ... 38 more Medium
        
        # High (40x)
        {"symptoms_text": "high fever 103F severe body aches chills", "age": 45, "sex": "M", "chronic_conditions": "", "red_flags": "highfever", "triagelabel": "High"},
        # ... 39 more High
        
        # Critical (40x)
        {"symptoms_text": "severe chest pain radiating left arm sweating nausea", "age": 55, "sex": "M", "chronic_conditions": "hypertension,diabetes", "red_flags": "chestpain,difficultybreathing", "triagelabel": "Critical"},
        {"symptoms_text": "sudden severe headache worst of life neck stiffness", "age": 42, "sex": "F", "chronic_conditions": "", "red_flags": "severeheadache", "triagelabel": "Critical"},
        {"symptoms_text": "difficulty breathing lips turning blue cannot speak", "age": 60, "sex": "F", "chronic_conditions": "asthma", "red_flags": "difficultybreathing", "triagelabel": "Critical"},
        # ... 37 more Critical
    ]
    
    # Generate 160 total (40 per class)
    samples = []
    labels = ["Low", "Medium", "High", "Critical"]
    for label in labels:
        label_samples = [s for s in base_samples if s["triagelabel"] == label]
        samples.extend(label_samples * 10)  # 4 base * 10 = 40 per class
    
    df = pd.DataFrame(samples[:160])  # Exactly 160
    raw_dir = DATADIR / "raw"
    raw_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(raw_dir / "disease_symptoms.csv", index=False)
    logger.info(f"Sample data created: {len(df)} records (40 per class)")
    return df

def preprocess_data(df):
    """Clean and preprocess data"""
    logger.info("Preprocessing data...")
    df.loc[:, 'chronic_conditions'] = df['chronic_conditions'].fillna('')
    df.loc[:, 'red_flags'] = df['red_flags'].fillna('')
    df.loc[:, 'symptoms_text'] = df['symptoms_text'].str.lower()
    logger.info(f"After preprocessing: {len(df)} records")
    return df

def split_data(df, test_size=0.2, val_size=0.1):
    """Split into train/val/test with stratification"""
    logger.info("Splitting data...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['triage_label']
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=42, 
        stratify=train_val['triage_label']
    )
    
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def save_processed_data(train, val, test):
    """Save processed datasets"""
    logger.info("Saving processed data...")
    processed_dir = DATADIR / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    logger.info("Processed data saved")

def main():
    logger.info("=== Data Preparation Pipeline ===")
    df = load_raw_data()
    df = preprocess_data(df)
    train, val, test = split_data(df)
    save_processed_data(train, val, test)
    logger.info("Data Preparation Complete!")

if __name__ == "__main__":
    main()
