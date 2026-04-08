import pandas as pd
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODEL_DIR, RESULTS_DIR
from src.data.feature_builder import FeatureBuilder
from src.models.ml_baselines import MLBaselines
from src.models.llm_council import LLMCouncil
from src.models.consensus_engine import ConsensusEngine
from src.inference.pipeline import InferencePipeline
from src.evaluation.metrics import TriageMetrics
from src.utils.logger import logger

async def evaluate_on_test_set():
    """Evaluate system on test set"""
    logger.info("=== System Evaluation ===\n")
    
    # Load models
    logger.info("Loading models...")
    fb = FeatureBuilder.load(MODEL_DIR / "feature_builder.pkl")
    ml = MLBaselines()
    ml.load(MODEL_DIR / "ml_baselines")
    llm = LLMCouncil()
    consensus = ConsensusEngine()
    pipeline = InferencePipeline(fb, ml, llm, consensus)
    
    # Load test data
    logger.info("Loading test data...")
    test_path = DATA_DIR / "processed" / "test.csv"
    test_df = pd.read_csv(test_path)
    logger.info(f"Test set size: {len(test_df)}\n")
    
    # Run predictions
    logger.info("Running predictions...")
    y_true = []
    y_pred = []
    
    for idx, row in test_df.iterrows():
        test_case = {
            'symptoms_text': row['symptoms_text'],
            'age': row['age'],
            'sex': row['sex'],
            'chronic_conditions': row['chronic_conditions'].split(',') if pd.notna(row['chronic_conditions']) and row['chronic_conditions'] else [],
            'red_flags': row['red_flags'].split(',') if pd.notna(row['red_flags']) and row['red_flags'] else []
        }
        
        result = await pipeline.run(test_case)
        
        if result['status'] == 'success':
            y_true.append(row['triage_label'])
            y_pred.append(result['risk_level'])
            
        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx + 1}/{len(test_df)} cases")
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics_calculator = TriageMetrics()
    metrics_calculator.print_report(y_true, y_pred)
    
    # Save results
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    results_path = RESULTS_DIR / "predictions" / "test_predictions.csv"
    results_path.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"\n✓ Results saved to {results_path}")
    logger.info("\n=== Evaluation Complete ===")

if __name__ == '__main__':
    asyncio.run(evaluate_on_test_set())
