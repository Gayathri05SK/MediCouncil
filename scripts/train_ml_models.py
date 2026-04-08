import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODEL_DIR
from src.data.feature_builder import FeatureBuilder
from src.models.ml_baselines import MLBaselines
from src.utils.logger import logger

def load_training_data():
    """Load processed training data"""
    logger.info("Loading training data...")
    
    train_path = DATA_DIR / "processed" / "train.csv"
    if not train_path.exists():
        logger.error("Training data not found. Run prepare_data.py first.")
        sys.exit(1)
    
    df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(df)} training samples\n")
    return df

def build_features(df):
    """Build features for training"""
    logger.info("Building features...")
    
    # Initialize and fit feature builder
    fb = FeatureBuilder()
    fb.fit(df['symptoms_text'].tolist())
    
    # Build features for all samples
    X_train = []
    for idx, row in df.iterrows():
        case = {
            'symptoms_text': row['symptoms_text'],
            'age': row['age'],
            'sex': row['sex'],
            'chronic_conditions': row['chronic_conditions'].split(',') if pd.notna(row['chronic_conditions']) and row['chronic_conditions'] else [],
            'red_flags': row['red_flags'].split(',') if pd.notna(row['red_flags']) and row['red_flags'] else []
        }
        features = fb.build_ml_features(case)
        X_train.append(features[0])
    
    X_train = np.array(X_train)
    y_train = df['triage_label'].values
    
    logger.info(f"Feature matrix shape: {X_train.shape}\n")
    
    return fb, X_train, y_train

def train_models(X_train, y_train):
    """Train ML baseline models"""
    logger.info("Training ML models...")
    
    ml = MLBaselines()
    ml.train(X_train, y_train)
    
    return ml

def save_models(feature_builder, ml_baselines):
    """Save trained models"""
    logger.info("Saving models...")
    
    # Save feature builder
    fb_path = MODEL_DIR / "feature_builder.pkl"
    feature_builder.save(fb_path)
    
    # Save ML models
    ml_path = MODEL_DIR / "ml_baselines"
    ml_baselines.save(ml_path)
    
    logger.info("✓ All models saved\n")

def main():
    """Main training pipeline"""
    logger.info("=== ML Model Training Pipeline ===\n")
    
    # Load data
    df = load_training_data()
    
    # Build features
    fb, X_train, y_train = build_features(df)
    
    # Train models
    ml = train_models(X_train, y_train)
    
    # Save models
    save_models(fb, ml)
    
    logger.info("=== Training Complete ===")

if __name__ == '__main__':
    main()
