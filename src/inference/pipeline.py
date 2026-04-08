import asyncio
from typing import Dict
from src.data.preprocessor import InputPreprocessor
from src.data.feature_builder import FeatureBuilder
from src.models.ml_baselines import MLBaselines
from src.models.llm_council import LLMCouncil
from src.models.consensus_engine import ConsensusEngine

class InferencePipeline:
    """Main inference pipeline orchestrator"""
    
    def __init__(
        self,
        feature_builder: FeatureBuilder,
        ml_baselines: MLBaselines,
        llm_council: LLMCouncil,
        consensus_engine: ConsensusEngine
    ):
        self.preprocessor = InputPreprocessor()
        self.feature_builder = feature_builder
        self.ml_baselines = ml_baselines
        self.llm_council = llm_council
        self.consensus_engine = consensus_engine
    
    async def run(self, raw_input: Dict) -> Dict:
        """Run complete inference pipeline"""
        try:
            # Stage 1: Validate and preprocess
            case = self.preprocessor.validate_and_clean(raw_input)
            
            # Stage 2: Feature engineering
            features = self.feature_builder.build_ml_features(case)
            
            # Stage 3: Parallel inference
            ml_predictions, ml_probabilities = self.ml_baselines.predict(features)
            llm_outputs = await self.llm_council.run_council(case)
            
            # Stage 4: Consensus
            decision = self.consensus_engine.compute_consensus(
                llm_outputs,
                ml_predictions,
                ml_probabilities
            )
            
            # Add metadata
            decision['input_case'] = case
            decision['status'] = 'success'
            
            return decision
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'message': 'An error occurred during triage. Please try again or seek medical advice.'
            }
    
    def run_sync(self, raw_input: Dict) -> Dict:
        """Synchronous wrapper for async run"""
        return asyncio.run(self.run(raw_input))
