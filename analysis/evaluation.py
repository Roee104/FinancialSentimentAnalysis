# analysis/evaluation.py
"""
Evaluation metrics for sentiment analysis results
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class SentimentEvaluator:
    """Handles evaluation of sentiment analysis results"""

    def __init__(self):
        """Initialize evaluator"""
        self.metrics = {}
        logger.info("Initialized SentimentEvaluator")

    def evaluate_predictions(self,
                             y_true: List[str],
                             y_pred: List[str],
                             name: str = "Model",
                             n_bootstrap: int = 100,
                             confidence_level: float = 0.95) -> Dict:
        """
        Evaluate sentiment predictions with bootstrap confidence intervals

        Args:
            y_true: True labels
            y_pred: Predicted labels
            name: Model name
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary of metrics with confidence intervals
        """
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        labels = ['Positive', 'Neutral', 'Negative']
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None
        )

        # Macro averages
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Bootstrap confidence intervals
        if n_bootstrap > 0:
            ci_metrics = self._bootstrap_metrics(
                y_true, y_pred, labels, n_bootstrap, confidence_level
            )
        else:
            ci_metrics = {}

        metrics = {
            'name': name,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class': {},
            'confidence_intervals': ci_metrics
        }

        # Per-class metrics
        for i, label in enumerate(labels):
            metrics['per_class'][label] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }

        metrics['confusion_matrix'] = cm.tolist()

        self.metrics[name] = metrics
        return metrics

    def _bootstrap_metrics(self,
                           y_true: List[str],
                           y_pred: List[str],
                           labels: List[str],
                           n_bootstrap: int,
                           confidence_level: float) -> Dict:
        """
        Calculate bootstrap confidence intervals for metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label list
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level

        Returns:
            Dict with confidence intervals
        """
        n_samples = len(y_true)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Store bootstrap results
        accuracies = []
        macro_f1s = []
        per_class_f1s = {label: [] for label in labels}

        # Bootstrap sampling
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = resample(range(n_samples), n_samples=n_samples)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Calculate metrics
            acc = accuracy_score(y_true_boot, y_pred_boot)
            accuracies.append(acc)

            _, _, f1, _ = precision_recall_fscore_support(
                y_true_boot, y_pred_boot, labels=labels, average=None, zero_division=0
            )
            macro_f1s.append(np.mean(f1))

            for i, label in enumerate(labels):
                per_class_f1s[label].append(f1[i])

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_results = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'ci_lower': np.percentile(accuracies, lower_percentile),
                'ci_upper': np.percentile(accuracies, upper_percentile)
            },
            'macro_f1': {
                'mean': np.mean(macro_f1s),
                'std': np.std(macro_f1s),
                'ci_lower': np.percentile(macro_f1s, lower_percentile),
                'ci_upper': np.percentile(macro_f1s, upper_percentile)
            },
            'per_class_f1': {}
        }

        for label in labels:
            f1_values = per_class_f1s[label]
            ci_results['per_class_f1'][label] = {
                'mean': np.mean(f1_values),
                'std': np.std(f1_values),
                'ci_lower': np.percentile(f1_values, lower_percentile),
                'ci_upper': np.percentile(f1_values, upper_percentile)
            }

        return ci_results

    def evaluate_against_gold_standard(self,
                                       predictions_file: Path,
                                       gold_standard_file: Path,
                                       name: str = "Model",
                                       n_bootstrap: int = 100) -> Dict:
        """
        Evaluate predictions against gold standard annotations

        Args:
            predictions_file: Path to predictions JSONL
            gold_standard_file: Path to gold standard JSONL
            name: Model name
            n_bootstrap: Number of bootstrap samples

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {name} against gold standard")

        # Load predictions
        predictions = {}
        with open(predictions_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Use article hash if available, otherwise use title
                key = data.get('article_hash', data.get('title'))
                predictions[key] = data.get('overall_sentiment')

        # Load gold standard
        gold_standard = {}
        with open(gold_standard_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                key = data.get('article_hash', data.get('title'))
                gold_standard[key] = data.get('true_overall')

        # Find common articles
        common_keys = set(predictions.keys()) & set(gold_standard.keys())
        logger.info(f"Found {len(common_keys)} articles in common")

        if not common_keys:
            logger.warning("No common articles found for evaluation")
            return {}

        # Extract aligned predictions and labels
        y_true = []
        y_pred = []

        for key in common_keys:
            y_true.append(gold_standard[key])
            y_pred.append(predictions[key])

        # Evaluate with bootstrap
        return self.evaluate_predictions(y_true, y_pred, name, n_bootstrap)

    def cross_validate(self,
                       data_file: Path,
                       pipeline_func,
                       n_folds: int = 5,
                       stratified: bool = True) -> Dict:
        """
        Perform cross-validation

        Args:
            data_file: Path to data file
            pipeline_func: Function that creates and runs pipeline
            n_folds: Number of folds
            stratified: Whether to use stratified splits

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold, KFold

        logger.info(f"Starting {n_folds}-fold cross-validation")

        # Load data
        df = pd.read_parquet(data_file)

        # Get labels
        if 'sentiment_label' in df.columns:
            labels = df['sentiment_label'].values
        else:
            logger.error("No sentiment labels found in data")
            return {}

        # Create folds
        if stratified:
            kfold = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df, labels)):
            logger.info(f"Processing fold {fold_idx + 1}/{n_folds}")

            # Split data
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Run pipeline on train and evaluate on test
            # This is a simplified version - actual implementation would need
            # to handle temporary files and pipeline execution
            fold_metrics = {
                'fold': fold_idx + 1,
                'train_size': len(train_df),
                'test_size': len(test_df)
            }

            fold_results.append(fold_metrics)

        return {
            'n_folds': n_folds,
            'fold_results': fold_results,
            'average_metrics': self._average_fold_metrics(fold_results)
        }

    def _average_fold_metrics(self, fold_results: List[Dict]) -> Dict:
        """Average metrics across folds"""
        # Simplified - would aggregate actual metrics
        return {
            'avg_train_size': np.mean([f['train_size'] for f in fold_results]),
            'avg_test_size': np.mean([f['test_size'] for f in fold_results])
        }

    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple model results

        Args:
            model_results: Dict mapping model names to metrics

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, metrics in model_results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0) * 100,
                'Macro F1': metrics.get('macro_f1', 0),
                'Positive F1': metrics['per_class']['Positive']['f1'],
                'Neutral F1': metrics['per_class']['Neutral']['f1'],
                'Negative F1': metrics['per_class']['Negative']['f1']
            }

            # Add confidence intervals if available
            if 'confidence_intervals' in metrics:
                ci = metrics['confidence_intervals']
                if 'accuracy' in ci:
                    row['Accuracy CI'] = f"[{ci['accuracy']['ci_lower']*100:.1f}, {ci['accuracy']['ci_upper']*100:.1f}]"
                if 'macro_f1' in ci:
                    row['Macro F1 CI'] = f"[{ci['macro_f1']['ci_lower']:.3f}, {ci['macro_f1']['ci_upper']:.3f}]"

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)

        return df

    def calculate_bias_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate bias-related metrics

        Args:
            results: List of result dictionaries

        Returns:
            Bias metrics
        """
        sentiments = [r.get('overall_sentiment', 'Unknown') for r in results]
        sentiment_counts = pd.Series(sentiments).value_counts()

        total = len(sentiments)
        neutral_pct = sentiment_counts.get('Neutral', 0) / total * 100

        # Calculate sentiment balance (deviation from ideal 33.3% each)
        ideal_pct = 33.33
        balance_score = 0

        for sentiment in ['Positive', 'Neutral', 'Negative']:
            actual_pct = sentiment_counts.get(sentiment, 0) / total * 100
            balance_score += abs(actual_pct - ideal_pct)

        # Lower score is better (0 = perfect balance)
        balance_score = 100 - balance_score

        return {
            'neutral_percentage': neutral_pct,
            'balance_score': balance_score,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'total_articles': total
        }

    def evaluate_ticker_extraction(self,
                                   results: List[Dict],
                                   expected_tickers: Optional[Dict] = None) -> Dict:
        """
        Evaluate ticker extraction performance

        Args:
            results: List of result dictionaries
            expected_tickers: Optional dict of expected tickers per article

        Returns:
            Ticker extraction metrics
        """
        total_articles = len(results)
        articles_with_tickers = sum(
            1 for r in results if r.get('ticker_count', 0) > 0)
        total_tickers = sum(r.get('ticker_count', 0) for r in results)

        metrics = {
            'coverage': articles_with_tickers / total_articles * 100 if total_articles > 0 else 0,
            'avg_tickers_per_article': total_tickers / total_articles if total_articles > 0 else 0,
            'articles_with_tickers': articles_with_tickers,
            'articles_without_tickers': total_articles - articles_with_tickers
        }

        # If we have expected tickers, calculate precision/recall
        if expected_tickers:
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for result in results:
                article_hash = result.get('article_hash', result.get('title'))
                extracted = set(t['symbol'] for t in result.get('tickers', []))
                expected = set(expected_tickers.get(article_hash, []))

                true_positives += len(extracted & expected)
                false_positives += len(extracted - expected)
                false_negatives += len(expected - extracted)

            precision = true_positives / \
                (true_positives + false_positives) if (true_positives +
                                                       false_positives) > 0 else 0
            recall = true_positives / \
                (true_positives + false_negatives) if (true_positives +
                                                       false_negatives) > 0 else 0
            f1 = 2 * precision * recall / \
                (precision + recall) if (precision + recall) > 0 else 0

            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        return metrics

    def generate_evaluation_report(self,
                                   results_files: Dict[str, Path],
                                   gold_standard_file: Optional[Path] = None,
                                   output_file: Optional[Path] = None) -> str:
        """
        Generate comprehensive evaluation report

        Args:
            results_files: Dict mapping model names to result files
            gold_standard_file: Optional gold standard for evaluation
            output_file: Optional path to save report

        Returns:
            Report string
        """
        report = []
        report.append("="*60)
        report.append("SENTIMENT ANALYSIS EVALUATION REPORT")
        report.append("="*60)

        all_metrics = {}

        for model_name, results_file in results_files.items():
            if not results_file.exists():
                continue

            # Load results
            results = []
            with open(results_file, 'r') as f:
                for line in f:
                    results.append(json.loads(line))

            report.append(f"\n{model_name}")
            report.append("-"*40)

            # Bias metrics
            bias_metrics = self.calculate_bias_metrics(results)
            report.append(
                f"Neutral bias: {bias_metrics['neutral_percentage']:.1f}%")
            report.append(
                f"Balance score: {bias_metrics['balance_score']:.1f}")

            # Ticker metrics
            ticker_metrics = self.evaluate_ticker_extraction(results)
            report.append(
                f"Ticker coverage: {ticker_metrics['coverage']:.1f}%")
            report.append(
                f"Avg tickers/article: {ticker_metrics['avg_tickers_per_article']:.2f}")

            # If gold standard available
            if gold_standard_file and gold_standard_file.exists():
                eval_metrics = self.evaluate_against_gold_standard(
                    results_file, gold_standard_file, model_name, n_bootstrap=100
                )
                if eval_metrics:
                    report.append(
                        f"Accuracy: {eval_metrics['accuracy']*100:.1f}%")
                    report.append(f"Macro F1: {eval_metrics['macro_f1']:.3f}")

                    # Add confidence intervals
                    if 'confidence_intervals' in eval_metrics:
                        ci = eval_metrics['confidence_intervals']['accuracy']
                        report.append(
                            f"Accuracy 95% CI: [{ci['ci_lower']*100:.1f}%, {ci['ci_upper']*100:.1f}%]")

                    all_metrics[model_name] = eval_metrics

        # Comparison table if multiple models evaluated
        if len(all_metrics) > 1:
            report.append("\n" + "="*60)
            report.append("MODEL COMPARISON")
            report.append("="*60)

            comparison_df = self.compare_models(all_metrics)
            report.append("\n" + comparison_df.to_string(index=False))

        report_text = "\n".join(report)

        # Save if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")

        return report_text


# Convenience functions
def evaluate_model(results_file: Path,
                   gold_standard_file: Path,
                   model_name: str = "Model",
                   n_bootstrap: int = 100) -> Dict:
    """Evaluate a single model against gold standard"""
    evaluator = SentimentEvaluator()
    return evaluator.evaluate_against_gold_standard(
        results_file, gold_standard_file, model_name, n_bootstrap
    )


def compare_all_models(results_dir: Path = None) -> pd.DataFrame:
    """Compare all models in results directory"""
    results_dir = results_dir or DATA_DIR

    evaluator = SentimentEvaluator()
    results_files = {
        "Standard Pipeline": results_dir / "processed_articles_standard.jsonl",
        "Optimized Pipeline": results_dir / "processed_articles_optimized.jsonl",
        "Calibrated Pipeline": results_dir / "processed_articles_calibrated.jsonl",
        "VADER Baseline": results_dir / "vader_baseline_results.jsonl"
    }

    # Remove non-existent files
    results_files = {k: v for k, v in results_files.items() if v.exists()}

    # Generate report
    report = evaluator.generate_evaluation_report(results_files)
    print(report)

    return evaluator.compare_models(evaluator.metrics)


if __name__ == "__main__":
    # Run evaluation
    compare_all_models()
