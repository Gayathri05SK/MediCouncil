import numpy as np
from typing import Dict, List, Tuple
from config.settings import CONSENSUS_CONFIG

class ConsensusEngine:
    """Safety-first consensus and decision logic"""
    
    def __init__(self):
        self.weights = CONSENSUS_CONFIG['weights']
        self.safety_threshold = CONSENSUS_CONFIG['safety_override_threshold']
        self.low_confidence_threshold = CONSENSUS_CONFIG['low_confidence_threshold']
        
        self.urgency_map = {
            'self_care': 0,
            'routine': 1,
            'urgent': 2,
            'emergency': 3
        }
        self.reverse_urgency_map = {v: k for k, v in self.urgency_map.items()}
    
    def compute_consensus(
        self,
        llm_outputs: Dict,
        ml_predictions: Dict = None,
        ml_probabilities: Dict = None
    ) -> Dict:
        """Compute final consensus decision"""
        
        # Safety override check
        safety_check = self._check_safety_override(llm_outputs)
        if safety_check['triggered']:
            return safety_check['decision']
        
        # Weighted consensus
        weighted_decision = self._weighted_consensus(llm_outputs)
        
        # Compute confidence
        confidence = self._compute_confidence(llm_outputs)
        
        # Generate explanation
        explanation = self._generate_explanation(llm_outputs, weighted_decision)
        
        # Final decision
        decision = {
            'risk_level': self._risk_score_to_level(weighted_decision['risk_score']),
            'risk_score': weighted_decision['risk_score'],
            'urgency': weighted_decision['urgency'],
            'confidence': confidence,
            'confidence_level': self._confidence_to_level(confidence),
            'explanation': explanation,
            'key_symptoms': self._merge_key_symptoms(llm_outputs),
            'agent_agreement': self._compute_agreement(llm_outputs),
            'safety_override': False,
            'disclaimer': self._get_disclaimer()
        }
        
        # Add ML baseline info if available
        if ml_predictions:
            decision['ml_baseline'] = {
                'predictions': ml_predictions,
                'probabilities': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in (ml_probabilities or {}).items()}
            }
        
        return decision
    
    def _check_safety_override(self, llm_outputs: Dict) -> Dict:
        """Check if safety override should trigger"""
        for agent_name, output in llm_outputs.items():
            if output.get('urgency') == 'emergency':
                risk_score = output.get('risk_score', 0)
                confidence = output.get('confidence', 0)
                
                # Override if emergency detected with sufficient confidence
                if risk_score >= 75 or confidence >= self.safety_threshold:
                    return {
                        'triggered': True,
                        'decision': {
                            'risk_level': 'Critical',
                            'risk_score': max(85, risk_score),
                            'urgency': 'emergency',
                            'confidence': max(0.8, confidence),
                            'confidence_level': 'High',
                            'explanation': f"SAFETY OVERRIDE: {agent_name.upper()} agent detected emergency condition. {output.get('reasoning', '')}",
                            'key_symptoms': output.get('key_symptoms', []),
                            'agent_agreement': self._compute_agreement(llm_outputs),
                            'safety_override': True,
                            'override_agent': agent_name,
                            'disclaimer': self._get_disclaimer(emergency=True)
                        }
                    }
        
        return {'triggered': False}
    
    def _weighted_consensus(self, llm_outputs: Dict) -> Dict:
        """Compute weighted consensus from agent outputs"""
        total_risk_score = 0
        total_urgency_score = 0
        total_weight = 0
        
        for agent_name, output in llm_outputs.items():
            if output.get('status') == 'success':
                weight = self.weights.get(agent_name, 0.33)
                
                # Weighted risk score
                risk = output.get('risk_score', 50)
                total_risk_score += risk * weight
                
                # Weighted urgency
                urgency = output.get('urgency', 'routine')
                urgency_score = self.urgency_map.get(urgency, 1)
                total_urgency_score += urgency_score * weight
                
                total_weight += weight
        
        if total_weight == 0:
            # Fallback if all agents failed
            return {'risk_score': 50, 'urgency': 'routine'}
        
        # Normalize
        final_risk_score = int(total_risk_score / total_weight)
        final_urgency_score = int(round(total_urgency_score / total_weight))
        final_urgency = self.reverse_urgency_map.get(final_urgency_score, 'routine')
        
        return {
            'risk_score': final_risk_score,
            'urgency': final_urgency
        }
    
    def _compute_confidence(self, llm_outputs: Dict) -> float:
        """Compute confidence based on agent agreement"""
        urgency_values = []
        confidences = []
        
        for output in llm_outputs.values():
            if output.get('status') == 'success':
                urgency_values.append(self.urgency_map.get(output.get('urgency', 'routine'), 1))
                confidences.append(output.get('confidence', 0.5))
        
        if not urgency_values:
            return 0.3
        
        # Agreement-based confidence
        unique_urgencies = len(set(urgency_values))
        if unique_urgencies == 1:
            agreement_confidence = 0.9  # All agree
        elif unique_urgencies == 2:
            agreement_confidence = 0.6  # Partial agreement
        else:
            agreement_confidence = 0.4  # Disagreement
        
        # Average agent confidence
        avg_agent_confidence = np.mean(confidences)
        
        # Combined confidence
        final_confidence = (agreement_confidence * 0.6 + avg_agent_confidence * 0.4)
        
        return round(final_confidence, 2)
    
    def _compute_agreement(self, llm_outputs: Dict) -> Dict:
        """Compute inter-agent agreement metrics"""
        urgency_values = [
            self.urgency_map.get(output.get('urgency', 'routine'), 1)
            for output in llm_outputs.values()
            if output.get('status') == 'success'
        ]
        
        if len(urgency_values) < 2:
            return {'rate': 0.0, 'level': 'unknown'}
        
        # Exact agreement rate
        most_common = max(set(urgency_values), key=urgency_values.count)
        agreement_rate = urgency_values.count(most_common) / len(urgency_values)
        
        return {
            'rate': round(agreement_rate, 2),
            'level': 'high' if agreement_rate >= 0.75 else 'medium' if agreement_rate >= 0.5 else 'low'
        }
    
    def _generate_explanation(self, llm_outputs: Dict, decision: Dict) -> str:
        """Generate combined explanation"""
        explanations = []
        
        for agent_name, output in llm_outputs.items():
            if output.get('status') == 'success' and output.get('reasoning'):
                explanations.append(f"**{agent_name.title()}**: {output['reasoning']}")
        
        if not explanations:
            return "Unable to generate detailed explanation."
        
        combined = "\n\n".join(explanations)
        
        # Add final recommendation
        urgency = decision['urgency']
        if urgency == 'emergency':
            recommendation = "\n\n**RECOMMENDATION**: Seek immediate emergency medical attention."
        elif urgency == 'urgent':
            recommendation = "\n\n**RECOMMENDATION**: See a healthcare provider within 24 hours."
        elif urgency == 'routine':
            recommendation = "\n\n**RECOMMENDATION**: Schedule a routine appointment with your doctor."
        else:
            recommendation = "\n\n**RECOMMENDATION**: Monitor symptoms. Self-care measures may be sufficient."
        
        return combined + recommendation
    
    def _merge_key_symptoms(self, llm_outputs: Dict) -> List[str]:
        """Merge key symptoms from all agents"""
        all_symptoms = set()
        for output in llm_outputs.values():
            if output.get('status') == 'success':
                symptoms = output.get('key_symptoms', [])
                all_symptoms.update(symptoms)
        
        return list(all_symptoms)[:5]  # Top 5
    
    def _risk_score_to_level(self, score: int) -> str:
        """Convert risk score to level"""
        if score >= 75:
            return 'Critical'
        elif score >= 50:
            return 'High'
        elif score >= 25:
            return 'Medium'
        else:
            return 'Low'
    
    def _confidence_to_level(self, confidence: float) -> str:
        """Convert confidence to level"""
        if confidence >= 0.7:
            return 'High'
        elif confidence >= 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_disclaimer(self, emergency: bool = False) -> str:
        """Get medical disclaimer"""
        base = "This is a decision support tool only, not a replacement for professional medical advice. "
        
        if emergency:
            return base + "IN CASE OF EMERGENCY, CALL YOUR LOCAL EMERGENCY NUMBER IMMEDIATELY (e.g., 911, 108)."
        else:
            return base + "Always consult with a qualified healthcare provider for proper diagnosis and treatment."
