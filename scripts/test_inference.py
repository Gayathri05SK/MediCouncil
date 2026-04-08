import asyncio
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import MODEL_DIR
from src.data.feature_builder import FeatureBuilder
from src.models.ml_baselines import MLBaselines
from src.models.llm_council import LLMCouncil
from src.models.consensus_engine import ConsensusEngine
from src.inference.pipeline import InferencePipeline
from src.utils.logger import logger

def load_models():
    """Load trained models"""
    logger.info("Loading models...")
    
    # Feature builder
    fb = FeatureBuilder.load(MODEL_DIR / "feature_builder.pkl")
    
    # ML baselines
    ml = MLBaselines()
    ml.load(MODEL_DIR / "ml_baselines")
    
    # LLM council
    llm = LLMCouncil()
    
    # Consensus engine
    consensus = ConsensusEngine()
    
    logger.info("✓ Models loaded\n")
    
    return fb, ml, llm, consensus

def get_test_cases():
    """Define test cases"""
    return [
        {
            'symptoms_text': 'severe chest pain radiating to left arm with sweating and nausea',
            'age': 55,
            'sex': 'M',
            'chronic_conditions': ['hypertension', 'diabetes'],
            'red_flags': ['chest_pain', 'difficulty_breathing']
        },
        {
            'symptoms_text': 'mild headache and runny nose for 2 days',
            'age': 28,
            'sex': 'F',
            'chronic_conditions': [],
            'red_flags': []
        },
        {
            'symptoms_text': 'high fever 103F with severe body aches and chills',
            'age': 42,
            'sex': 'M',
            'chronic_conditions': [],
            'red_flags': ['high_fever']
        }
    ]

async def test_pipeline():
    """Test the inference pipeline"""
    logger.info("=== Testing Inference Pipeline ===\n")
    
    # Load models
    fb, ml, llm, consensus = load_models()
    
    # Initialize pipeline
    pipeline = InferencePipeline(fb, ml, llm, consensus)
    
    # Test cases
    test_cases = get_test_cases()
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST CASE {i}")
        logger.info(f"{'='*60}")
        logger.info(f"Symptoms: {test_case['symptoms_text']}")
        logger.info(f"Age: {test_case['age']}, Sex: {test_case['sex']}")
        logger.info(f"Red flags: {test_case.get('red_flags', [])}\n")
        
        # Run inference
        result = await pipeline.run(test_case)
        
        # Display results
        if result['status'] == 'success':
            logger.info("RESULTS:")
            logger.info(f"  Risk Level: {result['risk_level']}")
            logger.info(f"  Risk Score: {result['risk_score']}/100")
            logger.info(f"  Urgency: {result['urgency'].upper()}")
            logger.info(f"  Confidence: {result['confidence']} ({result['confidence_level']})")
            logger.info(f"  Safety Override: {result['safety_override']}")
            logger.info(f"  Agent Agreement: {result['agent_agreement']['rate']} ({result['agent_agreement']['level']})")
            logger.info(f"\nExplanation:\n{result['explanation']}")
        else:
            logger.error(f"ERROR: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n{'='*60}")
    logger.info("=== Testing Complete ===")

if __name__ == '__main__':
    asyncio.run(test_pipeline())
