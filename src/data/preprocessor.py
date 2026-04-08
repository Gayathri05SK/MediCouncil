import re
from typing import Dict, List, Optional

class InputPreprocessor:
    """Validates and normalizes user input"""
    
    VALID_RED_FLAGS = [
        'chest_pain', 'difficulty_breathing', 'severe_bleeding',
        'loss_consciousness', 'stroke_signs', 'severe_headache',
        'high_fever', 'severe_abdominal_pain', 'confusion',
        'suicidal_thoughts'
    ]
    
    def __init__(self):
        pass
    
    def validate_and_clean(self, raw_input: Dict) -> Dict:
        """Validate and clean raw input"""
        
        # Required field check
        if 'symptoms_text' not in raw_input or not raw_input['symptoms_text']:
            raise ValueError("symptoms_text is required")
        
        # Clean and structure
        cleaned = {
            'symptoms_text': self._clean_text(raw_input['symptoms_text']),
            'age': self._parse_age(raw_input.get('age')),
            'sex': self._parse_sex(raw_input.get('sex')),
            'chronic_conditions': self._parse_list(raw_input.get('chronic_conditions', [])),
            'red_flags': self._validate_red_flags(raw_input.get('red_flags', []))
        }
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean symptom text"""
        # Lowercase
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical terms
        text = re.sub(r'[^a-z0-9\s\-/]', '', text)
        return text
    
    def _parse_age(self, age) -> int:
        """Parse and validate age"""
        try:
            age = int(age)
            if 0 <= age <= 120:
                return age
            return 0  # Unknown
        except (ValueError, TypeError):
            return 0
    
    def _parse_sex(self, sex) -> str:
        """Parse and validate sex"""
        if isinstance(sex, str):
            sex = sex.upper().strip()
            if sex in ['M', 'F', 'MALE', 'FEMALE']:
                return 'M' if sex.startswith('M') else 'F'
        return 'U'  # Unknown
    
    def _parse_list(self, items) -> List[str]:
        """Parse list input"""
        if isinstance(items, str):
            items = [i.strip() for i in items.split(',') if i.strip()]
        elif not isinstance(items, list):
            items = []
        return [self._clean_text(item) for item in items]
    
    def _validate_red_flags(self, flags) -> List[str]:
        """Validate red flags against approved list"""
        flags = self._parse_list(flags)
        return [f for f in flags if f in self.VALID_RED_FLAGS]
