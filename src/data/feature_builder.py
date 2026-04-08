import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict

class FeatureBuilder:
    """Converts case to ML feature vector"""
    
    def __init__(self, max_features=500, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        self.fitted = False
    
    def fit(self, symptom_texts: list):
        """Fit vectorizer on training data"""
        self.vectorizer.fit(symptom_texts)
        self.fitted = True
        return self
    
    def build_ml_features(self, case: Dict) -> np.ndarray:
        """Build feature vector from case"""
        if not self.fitted:
            raise RuntimeError("FeatureBuilder not fitted. Call fit() first.")
        
        # Text features
        text_features = self.vectorizer.transform([case['symptoms_text']]).toarray()
        
        # Structured features
        age_group = self._encode_age_group(case.get('age', 0))
        sex_encoded = 1 if case.get('sex') == 'M' else 0
        has_chronic = 1 if case.get('chronic_conditions', []) else 0
        red_flag_count = len(case.get('red_flags', []))
        
        structured = np.array([[age_group, sex_encoded, has_chronic, red_flag_count]])
        
        # Combine
        features = np.hstack([text_features, structured])
        return features
    
    def _encode_age_group(self, age: int) -> int:
        """Encode age into groups"""
        if age < 18:
            return 0  # Pediatric
        elif age <= 40:
            return 1  # Young adult
        elif age <= 65:
            return 2  # Middle-aged
        else:
            return 3  # Elderly
    
    def save(self, path: str):
        """Save feature builder"""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str):
        """Load feature builder"""
        return joblib.load(path)
