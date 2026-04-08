from typing import Dict, List
import re

class InputValidator:
    """Validate user inputs"""
    
    @staticmethod
    def validate_symptoms(symptoms_text: str) -> bool:
        """Check if symptoms text is valid"""
        if not symptoms_text or len(symptoms_text.strip()) < 3:
            return False
        if len(symptoms_text) > 2000:
            return False
        return True
    
    @staticmethod
    def validate_age(age) -> bool:
        """Validate age"""
        try:
            age = int(age)
            return 0 <= age <= 120
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_sex(sex: str) -> bool:
        """Validate sex"""
        if isinstance(sex, str):
            return sex.upper() in ['M', 'F', 'MALE', 'FEMALE', 'U', 'UNKNOWN']
        return False
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input"""
        # Remove potential script injections
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<.*?>', '', text)
        return text.strip()
