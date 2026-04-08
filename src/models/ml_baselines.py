import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from typing import Dict, Tuple

class MLBaselines:
    """Classical ML models for benchmarking"""
    
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
        }
        self.trained = False
        self.label_encoder = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all baseline models"""
        print("Training ML Baselines...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
        
        self.trained = True
        print("✓ All models trained\n")
    
    def predict(self, X: np.ndarray) -> Tuple[Dict, Dict]:
        """Get predictions from all models"""
        if not self.trained:
            raise RuntimeError("Models not trained. Call train() first.")
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)[0]
            if hasattr(model, 'predict_proba'):
                probabilities[name] = model.predict_proba(X)[0]
        
        return predictions, probabilities
    
    def predict_batch(self, X: np.ndarray) -> Tuple[Dict, Dict]:
        """Batch predictions"""
        if not self.trained:
            raise RuntimeError("Models not trained. Call train() first.")
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
            if hasattr(model, 'predict_proba'):
                probabilities[name] = model.predict_proba(X)
        
        return predictions, probabilities
    
    def save(self, path: Path):
        """Save trained models"""
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        
        for name, model in self.models.items():
            joblib.dump(model, path / f'{name}.pkl')
        
        print(f"✓ Models saved to {path}")
    
    def load(self, path: Path):
        """Load trained models"""
        path = Path(path)
        
        for name in self.models.keys():
            model_path = path / f'{name}.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.models[name] = joblib.load(model_path)
        
        self.trained = True
        print(f"✓ Models loaded from {path}")
