import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)
from typing import Dict, List

class TriageMetrics:
    """Evaluation metrics for triage system"""
    
    def __init__(self, classes=['Low', 'Medium', 'High', 'Critical']):
        self.classes = classes
    
    def compute_all_metrics(self, y_true: List, y_pred: List) -> Dict:
        """Compute all evaluation metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.classes, zero_division=0
        )
        
        metrics['per_class'] = {}
        for i, cls in enumerate(self.classes):
            metrics['per_class'][cls] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
        
        # Macro/weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics['macro'] = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
        
        metrics['weighted'] = {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Critical metrics (safety focus)
        critical_idx = self.classes.index('Critical') if 'Critical' in self.classes else -1
        if critical_idx >= 0:
            critical_recall = recall[critical_idx]
            metrics['critical_recall'] = critical_recall
            
            # False negative rate for Critical
            critical_mask = np.array(y_true) == 'Critical'
            if critical_mask.sum() > 0:
                fn_rate = 1 - critical_recall
                metrics['critical_fn_rate'] = fn_rate
        
        return metrics
    
    def print_report(self, y_true: List, y_pred: List):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, labels=self.classes, zero_division=0))
        
        metrics = self.compute_all_metrics(y_true, y_pred)
        
        print("\n" + "="*60)
        print("KEY METRICS")
        print("="*60)
        print(f"Accuracy:          {metrics['accuracy']:.3f}")
        print(f"Macro F1:          {metrics['macro']['f1']:.3f}")
        print(f"Weighted F1:       {metrics['weighted']['f1']:.3f}")
        print(f"Cohen's Kappa:     {metrics['cohen_kappa']:.3f}")
        
        if 'critical_recall' in metrics:
            print(f"\nCritical Recall:   {metrics['critical_recall']:.3f} (SAFETY CRITICAL)")
            print(f"Critical FN Rate:  {metrics.get('critical_fn_rate', 0):.3f}")
