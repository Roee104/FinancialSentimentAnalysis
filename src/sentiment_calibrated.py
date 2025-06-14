# src/sentiment_calibrated.py

"""
Fixed calibrated sentiment analysis to properly reduce neutral bias.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Import base sentiment analyzer
from sentiment import FinBERTSentimentAnalyzer

class FixedCalibratedFinBERTAnalyzer(FinBERTSentimentAnalyzer):
    """
    Fixed Enhanced FinBERT that properly reduces neutral bias
    """
    
    def __init__(self, 
                 model_name: str = "yiyanghkust/finbert-tone",
                 device: str = None,
                 neutral_penalty: float = 0.7,  # Reduce neutral probability
                 pos_boost: float = 1.3,        # Boost positive signals
                 neg_boost: float = 1.3,        # Boost negative signals
                 min_confidence_diff: float = 0.05):  # Minimum difference to classify as non-neutral
        super().__init__(model_name, device)
        self.neutral_penalty = neutral_penalty
        self.pos_boost = pos_boost
        self.neg_boost = neg_boost
        self.min_confidence_diff = min_confidence_diff
        
        # Track calibration statistics
        self.calibration_stats = defaultdict(int)
        
    def _reduce_neutral_bias(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Apply bias reduction to counter excessive neutral predictions
        """
        adjusted_probs = probs.clone()
        
        # Reduce neutral class probability
        adjusted_probs[:, 1] *= self.neutral_penalty  # Neutral class
        
        # Boost positive and negative classes
        adjusted_probs[:, 0] *= self.neg_boost  # Negative class
        adjusted_probs[:, 2] *= self.pos_boost  # Positive class
        
        # Renormalize probabilities
        adjusted_probs = F.softmax(adjusted_probs, dim=1)
        
        return adjusted_probs
    
    def _classify_with_confidence_threshold(self, probs: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Classify with minimum confidence difference requirement
        """
        results = []
        
        for i in range(probs.size(0)):
            prob_neg = probs[i][0].item()
            prob_neu = probs[i][1].item()
            prob_pos = probs[i][2].item()
            
            # Sort probabilities to find top 2
            sorted_probs = sorted([
                ('Negative', prob_neg),
                ('Neutral', prob_neu),
                ('Positive', prob_pos)
            ], key=lambda x: x[1], reverse=True)
            
            top_label, top_prob = sorted_probs[0]
            second_label, second_prob = sorted_probs[1]
            
            # Require minimum confidence difference for non-neutral classification
            confidence_diff = top_prob - second_prob
            
            if top_label != 'Neutral' and confidence_diff >= self.min_confidence_diff:
                # Strong enough signal for positive/negative
                final_label = top_label
                final_confidence = top_prob
                self.calibration_stats[f'{final_label.lower()}_classified'] += 1
            elif second_label != 'Neutral' and (top_prob - prob_neu) >= self.min_confidence_diff:
                # Second choice is non-neutral and significantly better than neutral
                final_label = second_label if sorted_probs[1][1] > prob_neu else 'Neutral'
                final_confidence = sorted_probs[1][1] if final_label != 'Neutral' else prob_neu
                if final_label != 'Neutral':
                    self.calibration_stats[f'{final_label.lower()}_classified'] += 1
                else:
                    self.calibration_stats['neutral_classified'] += 1
            else:
                # Default to neutral
                final_label = 'Neutral'
                final_confidence = prob_neu
                self.calibration_stats['neutral_classified'] += 1
            
            results.append((final_label, final_confidence))
        
        return results
    
    def predict_calibrated(self,
                          texts: List[str],
                          batch_size: int = 16,
                          max_length: int = 512) -> List[Dict]:
        """
        Predict with bias reduction and confidence thresholding
        """
        if not texts:
            return []
            
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch,
                                padding=True,
                                truncation=True,
                                max_length=max_length,
                                return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask)
                
                # Get original probabilities
                original_probs = F.softmax(outputs.logits, dim=-1)
                
                # Apply bias reduction
                adjusted_probs = self._reduce_neutral_bias(original_probs)
                
                # Classify with confidence thresholding
                classifications = self._classify_with_confidence_threshold(adjusted_probs)
                
                # Build results
                for idx, (text, (label, confidence)) in enumerate(zip(batch, classifications)):
                    # Get all probabilities for reference
                    prob_neg = adjusted_probs[idx][0].item()
                    prob_neu = adjusted_probs[idx][1].item()
                    prob_pos = adjusted_probs[idx][2].item()
                    
                    # Track original prediction for comparison
                    original_max_idx = torch.argmax(original_probs[idx]).item()
                    original_label = ["Negative", "Neutral", "Positive"][original_max_idx]
                    
                    if original_label != label:
                        self.calibration_stats[f'changed_{original_label.lower()}_to_{label.lower()}'] += 1
                    
                    score_dict = {
                        "Negative": prob_neg,
                        "Neutral": prob_neu, 
                        "Positive": prob_pos
                    }
                    
                    results.append({
                        "text": text,
                        "label": label,
                        "confidence": confidence,
                        "scores": score_dict,
                        "original_label": original_label,
                        "calibrated": original_label != label
                    })
        
        return results
    
    def get_calibration_stats(self) -> Dict[str, int]:
        """Get calibration statistics"""
        return dict(self.calibration_stats)
    
    def reset_calibration_stats(self):
        """Reset calibration statistics"""
        self.calibration_stats.clear()

def test_bias_reduction_effectiveness():
    """Test different calibration settings to find optimal bias reduction"""
    
    # Sample texts with varying sentiment intensities
    test_texts = [
        "The company reported quarterly earnings that slightly exceeded expectations.",
        "Revenue growth was modest compared to the previous quarter's performance.",
        "Stock price remained relatively stable throughout the trading session.",
        "The CEO expressed cautious optimism about future market conditions.",
        "Analysts noted mixed signals in the company's financial statements.",
        "The quarterly dividend payment was maintained at current levels.",
        "Management provided updated guidance for the upcoming fiscal year.",
        "Market reaction to the earnings announcement was largely muted.",
        "The company continues to navigate challenging industry conditions.",
        "Several business segments showed incremental improvement this quarter."
    ]
    
    print("🔧 Testing Bias Reduction Effectiveness")
    print("=" * 50)
    
    # Test different calibration settings
    settings = [
        {"name": "Original", "params": None},
        {"name": "Light Calibration", "params": {"neutral_penalty": 0.9, "pos_boost": 1.1, "neg_boost": 1.1}},
        {"name": "Medium Calibration", "params": {"neutral_penalty": 0.8, "pos_boost": 1.2, "neg_boost": 1.2}},
        {"name": "Strong Calibration", "params": {"neutral_penalty": 0.7, "pos_boost": 1.3, "neg_boost": 1.3}},
        {"name": "Aggressive Calibration", "params": {"neutral_penalty": 0.6, "pos_boost": 1.4, "neg_boost": 1.4}},
    ]
    
    results = {}
    
    for setting in settings:
        print(f"\nTesting {setting['name']}...")
        
        if setting["params"] is None:
            # Original FinBERT
            analyzer = FinBERTSentimentAnalyzer()
            predictions = analyzer.predict(test_texts)
        else:
            # Calibrated FinBERT
            analyzer = FixedCalibratedFinBERTAnalyzer(**setting["params"])
            predictions = analyzer.predict_calibrated(test_texts)
        
        # Calculate distribution
        distribution = _get_distribution(predictions)
        results[setting["name"]] = distribution
        
        print(f"  Positive: {distribution.get('Positive', 0):.1f}%")
        print(f"  Neutral:  {distribution.get('Neutral', 0):.1f}%")
        print(f"  Negative: {distribution.get('Negative', 0):.1f}%")
    
    return results

def _get_distribution(results: List[Dict]) -> Dict[str, float]:
    """Get percentage distribution of sentiment labels"""
    if not results:
        return {}
        
    counts = defaultdict(int)
    for result in results:
        counts[result["label"]] += 1
    
    total = len(results)
    return {label: count/total*100 for label, count in counts.items()}

def find_optimal_calibration(texts: List[str]) -> Dict:
    """Find optimal calibration parameters for given texts"""
    
    print("🎯 Finding Optimal Calibration Parameters")
    print("=" * 50)
    
    # Get original distribution
    original_analyzer = FinBERTSentimentAnalyzer()
    original_results = original_analyzer.predict(texts)
    original_dist = _get_distribution(original_results)
    
    print(f"Original distribution: {original_dist}")
    original_neutral = original_dist.get('Neutral', 0)
    
    # Test different parameter combinations
    best_params = None
    best_neutral_reduction = 0
    best_distribution = None
    
    penalty_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    boost_values = [1.1, 1.2, 1.3, 1.4, 1.5]
    
    print("\nTesting parameter combinations...")
    
    for penalty in penalty_values:
        for boost in boost_values:
            try:
                analyzer = FixedCalibratedFinBERTAnalyzer(
                    neutral_penalty=penalty,
                    pos_boost=boost,
                    neg_boost=boost,
                    min_confidence_diff=0.05
                )
                
                results = analyzer.predict_calibrated(texts)
                distribution = _get_distribution(results)
                neutral_pct = distribution.get('Neutral', 0)
                
                neutral_reduction = original_neutral - neutral_pct
                
                # Look for significant neutral reduction without going too extreme
                if (neutral_reduction > best_neutral_reduction and 
                    neutral_pct > 15 and  # Don't eliminate neutral completely
                    neutral_pct < 70):    # Should still reduce from original
                    
                    best_neutral_reduction = neutral_reduction
                    best_params = {"neutral_penalty": penalty, "pos_boost": boost, "neg_boost": boost}
                    best_distribution = distribution
                    
            except Exception as e:
                print(f"Error with penalty={penalty}, boost={boost}: {e}")
                continue
    
    if best_params:
        print(f"\n✅ Best parameters found:")
        print(f"   Neutral penalty: {best_params['neutral_penalty']}")
        print(f"   Boost factor: {best_params['pos_boost']}")
        print(f"   Neutral reduction: {best_neutral_reduction:.1f} percentage points")
        print(f"   Final distribution: {best_distribution}")
    else:
        print("❌ No optimal parameters found")
    
    return {
        "best_params": best_params,
        "best_distribution": best_distribution,
        "neutral_reduction": best_neutral_reduction,
        "original_distribution": original_dist
    }

if __name__ == "__main__":
    # Test bias reduction approaches
    print("🧪 Testing Fixed Calibrated Sentiment Analysis")
    print("=" * 50)
    
    # Test different settings
    bias_results = test_bias_reduction_effectiveness()
    
    # Sample texts for optimization
    sample_texts = [
        "Apple reported strong quarterly earnings with revenue beating expectations by 15%.",
        "The company faced significant challenges due to supply chain disruptions.",
        "Stock prices remained relatively stable throughout the trading session.",
        "Investors are optimistic about the company's future growth prospects.",
        "Concerns about regulatory changes are weighing on market sentiment.",
        "The quarterly report shows mixed results with some positive indicators.",
        "Microsoft cloud services revenue exceeded forecasts this quarter.",
        "Tesla delivery numbers surpassed expectations despite challenges.",
        "Bank earnings were solid with loan growth offsetting provisions.",
        "Pharmaceutical companies announced positive trial results recently."
    ]
    
    # Find optimal calibration
    optimization_results = find_optimal_calibration(sample_texts)
    
    print("\n✅ Fixed calibrated sentiment analysis testing completed!")