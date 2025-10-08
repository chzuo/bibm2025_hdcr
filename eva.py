#!/usr/bin/env python3
"""
Unified Evaluation System - Read prediction JSON files and generate evaluation reports

Usage:
python unified_evaluation.py --input predictions.json --output evaluation_results.json
python unified_evaluation.py --input predictions.json --visualize --language
python unified_evaluation.py --compare model1_predictions.json model2_predictions.json --language
python unified_evaluation.py --latex model1.json model2.json model3.json
"""

import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedEvaluator:
    """Unified evaluator with cross-lingual support"""
    
    def __init__(self, label_names: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            label_names: List of label names, use default if None
        """
        self.label_names = label_names or [
            'none',
            'over_generalization',
            'improper_restriction',
            'effect_exaggeration',
            'spurious_causation'
        ]
        self.n_distortion_types = len(self.label_names) - 1
    
    def load_predictions(self, prediction_file: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load predictions from JSON file
        
        Args:
            prediction_file: Path to prediction JSON file
            
        Returns:
            y_true: True labels array
            y_pred: Predicted labels array
            predictions: Original prediction details list
        """
        logger.info(f"Loading predictions: {prediction_file}")
        
        with open(prediction_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract prediction details
        predictions = data.get('predictions', [])
        
        # Extract true and predicted labels
        y_true = np.array([p['true_label'] for p in predictions])
        y_pred = np.array([p['pred_label'] for p in predictions])
        
        logger.info(f"Loaded {len(predictions)} predictions")
        
        # Check for language information
        has_language = all('language' in p for p in predictions[:10])
        if has_language:
            languages = [p.get('language', 'unknown') for p in predictions]
            language_counts = pd.Series(languages).value_counts()
            logger.info(f"Language distribution: {language_counts.to_dict()}")
        
        return y_true, y_pred, predictions
    
    def evaluate_all(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     predictions: List[Dict] = None, model_info: Dict = None,
                     evaluate_language: bool = True) -> Dict:
        """
        Execute complete evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            predictions: Original predictions list (with language info)
            model_info: Model information (optional)
            evaluate_language: Whether to perform language evaluation
            
        Returns:
            Dictionary containing all evaluation results
        """
        results = {}
        
        # Add model information
        if model_info:
            results['model_info'] = model_info
        
        # ========== Overall Evaluation ==========
        logger.info("\n" + "="*60)
        logger.info("Overall Performance Evaluation")
        logger.info("="*60)
        
        # Binary classification evaluation
        logger.info("\nBinary Classification: Detect misinformation presence")
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        binary_results = self._evaluate_binary(y_true_binary, y_pred_binary)
        results['binary_classification'] = binary_results
        
        # 4-class evaluation (distorted samples only)
        logger.info(f"\n4-class Classification: Evaluate distorted samples only")
        mask_multi = y_true > 0
        if mask_multi.any():
            y_true_multi = y_true[mask_multi]
            y_pred_multi = y_pred[mask_multi]
            
            multi_class_results = self._evaluate_multiclass(
                y_true_multi, y_pred_multi, 
                class_names=self.label_names[1:],
                task_name="4-class"
            )
        else:
            multi_class_results = self._create_empty_multiclass_results()
        
        results['4_class_classification'] = multi_class_results
        
        # 5-class full evaluation
        logger.info(f"\n5-class Classification: All samples")
        full_class_results = self._evaluate_multiclass(
            y_true, y_pred,
            class_names=self.label_names,
            task_name="5-class"
        )
        results['5_class_classification'] = full_class_results
        
        # ========== Language Evaluation ==========
        if evaluate_language and predictions:
            languages = [p.get('language', 'unknown') for p in predictions]
            if 'unknown' not in languages or languages.count('unknown') < len(languages) * 0.5:
                logger.info("\n" + "="*60)
                logger.info("Cross-lingual Evaluation")
                logger.info("="*60)
                results['language_evaluation'] = self._evaluate_by_language(
                    y_true, y_pred, languages, predictions
                )
        
        # Summary statistics
        results['summary'] = self._create_summary(results)
        
        return results
    
    def _evaluate_by_language(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         languages: List[str], predictions: List[Dict]) -> Dict:
        """Evaluate performance by language"""
        language_results = {}
        unique_languages = list(set(languages))
        
        logger.info(f"Found {len(unique_languages)} languages: {unique_languages}")
        
        for lang in unique_languages:
            # Create language mask
            lang_mask = np.array([l == lang for l in languages])
            n_samples = int(lang_mask.sum())
            
            if n_samples < 5:  # Skip if too few samples
                logger.warning(f"Language {lang} has too few samples ({n_samples}), skipping")
                continue
            
            logger.info(f"\nEvaluating language: {lang} ({n_samples} samples)")
            
            # Get language subset
            y_true_lang = y_true[lang_mask]
            y_pred_lang = y_pred[lang_mask]
            
            # Binary classification
            y_true_binary = (y_true_lang > 0).astype(int)
            y_pred_binary = (y_pred_lang > 0).astype(int)
            
            # Calculate binary confusion matrix
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
                if 0 in y_true_binary and 0 in y_pred_binary:
                    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
                if 0 in y_true_binary and 1 in y_pred_binary:
                    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
                if 1 in y_true_binary and 0 in y_pred_binary:
                    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
                if 1 in y_true_binary and 1 in y_pred_binary:
                    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
            
            binary_metrics = {
                'accuracy': float(accuracy_score(y_true_binary, y_pred_binary)),
                'f1_score': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                'balanced_accuracy': float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
                'mcc': float(matthews_corrcoef(y_true_binary, y_pred_binary)),
                'confusion_matrix': {
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                }
            }
            
            # Multiclass evaluation
            multiclass_metrics = {
                'accuracy': float(accuracy_score(y_true_lang, y_pred_lang)),
                'macro_f1': float(f1_score(y_true_lang, y_pred_lang, average='macro', zero_division=0)),
                'weighted_f1': float(f1_score(y_true_lang, y_pred_lang, average='weighted', zero_division=0))
            }
            
            # Per-class performance
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_lang, y_pred_lang, average=None, 
                labels=list(range(len(self.label_names))), zero_division=0
            )
            
            per_class_metrics = {}
            for i, label_name in enumerate(self.label_names):
                if i < len(support) and support[i] > 0:
                    per_class_metrics[label_name] = {
                        'precision': float(precision[i]),
                        'recall': float(recall[i]),
                        'f1_score': float(f1[i]),
                        'support': int(support[i])
                    }
            
            language_results[lang] = {
                'n_samples': int(n_samples),
                'language_name': 'English' if lang == 'en' else 'Chinese' if lang == 'cn' else lang,
                'binary_classification': binary_metrics,
                'multiclass_classification': multiclass_metrics,
                'per_class': per_class_metrics
            }
        
        # Print language comparison
        self._print_language_comparison(language_results)
        
        return language_results
    
    def _print_language_comparison(self, language_results: Dict):
        """Print language comparison results"""
        
        print("\n" + "="*80)
        print("Cross-lingual Performance Comparison")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        
        for lang_code, metrics in language_results.items():
            comparison_data.append({
                'Language': metrics['language_name'],
                'Samples': metrics['n_samples'],
                'Binary Acc': f"{metrics['binary_classification']['accuracy']:.4f}",
                'Binary F1': f"{metrics['binary_classification']['f1_score']:.4f}",
                'Binary MCC': f"{metrics['binary_classification']['mcc']:.4f}",
                'Multi Acc': f"{metrics['multiclass_classification']['accuracy']:.4f}",
                'Multi Macro F1': f"{metrics['multiclass_classification']['macro_f1']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
        
        # Calculate performance differences
        if len(language_results) >= 2:
            langs = list(language_results.keys())
            if 'en' in langs and 'cn' in langs:
                en_metrics = language_results['en']
                cn_metrics = language_results['cn']
                
                print("\nPerformance Difference Analysis (English vs Chinese):")
                
                # Binary F1 difference
                binary_f1_diff = en_metrics['binary_classification']['f1_score'] - cn_metrics['binary_classification']['f1_score']
                print(f"  Binary F1 Diff: {abs(binary_f1_diff):.4f} ({'EN better' if binary_f1_diff > 0 else 'CN better'})")
                
                # Multiclass F1 difference
                multi_f1_diff = en_metrics['multiclass_classification']['macro_f1'] - cn_metrics['multiclass_classification']['macro_f1']
                print(f"  Multi Macro F1 Diff: {abs(multi_f1_diff):.4f} ({'EN better' if multi_f1_diff > 0 else 'CN better'})")
                
                # Per-class differences
                print("\nPer-class F1 Score Differences:")
                for class_name in self.label_names:
                    if class_name in en_metrics['per_class'] and class_name in cn_metrics['per_class']:
                        en_f1 = en_metrics['per_class'][class_name]['f1_score']
                        cn_f1 = cn_metrics['per_class'][class_name]['f1_score']
                        diff = en_f1 - cn_f1
                        print(f"    {class_name}: EN={en_f1:.3f}, CN={cn_f1:.3f}, Diff={diff:+.3f}")
    
    def _create_empty_multiclass_results(self) -> Dict:
        """Create empty multiclass results"""
        return {
            'accuracy': 0,
            'balanced_accuracy': 0,
            'macro': {'precision': 0, 'recall': 0, 'f1_score': 0},
            'weighted': {'precision': 0, 'recall': 0, 'f1_score': 0},
            'per_class': {},
            'confusion_matrix': [],
            'total_samples': 0,
            'note': 'No positive samples found'
        }
    
    def _evaluate_binary(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate binary classification performance"""
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle case with only one class
            tn = fp = fn = tp = 0
            if 0 in y_true and 0 in y_pred:
                tn = np.sum((y_true == 0) & (y_pred == 0))
            if 0 in y_true and 1 in y_pred:
                fp = np.sum((y_true == 0) & (y_pred == 1))
            if 1 in y_true and 0 in y_pred:
                fn = np.sum((y_true == 1) & (y_pred == 0))
            if 1 in y_true and 1 in y_pred:
                tp = np.sum((y_true == 1) & (y_pred == 1))
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
        
        self._print_binary_results(metrics)
        
        return metrics
    
    def _evaluate_multiclass(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str], task_name: str) -> Dict:
        """Evaluate multiclass performance"""
        
        # Determine expected labels
        if task_name == "4-class":
            expected_labels = list(range(1, len(class_names) + 1))
        else:
            expected_labels = list(range(len(class_names)))
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=expected_labels, zero_division=0
        )
        
        # Calculate average metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', labels=expected_labels, zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', labels=expected_labels, zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'macro': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1_score': float(f1_macro)
            },
            'weighted': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1_score': float(f1_weighted)
            },
            'per_class': {},
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=expected_labels).tolist(),
            'total_samples': len(y_true)
        }
        
        # Detailed metrics per class
        for i, label_idx in enumerate(expected_labels):
            if task_name == "4-class":
                class_name = class_names[i]
            else:
                class_name = class_names[label_idx] if label_idx < len(class_names) else f'class_{label_idx}'
                
            metrics['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        self._print_multiclass_results(metrics, task_name)
        
        return metrics
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create summary statistics"""
        summary = {
            'overall_performance': {
                'binary_f1': results['binary_classification']['f1_score'],
                'binary_mcc': results['binary_classification']['matthews_corrcoef'],
                'multiclass_macro_f1': results['5_class_classification']['macro']['f1_score'],
                'multiclass_accuracy': results['5_class_classification']['accuracy']
            }
        }
        
        # Add language insights if available
        if 'language_evaluation' in results:
            lang_eval = results['language_evaluation']
            if 'en' in lang_eval and 'cn' in lang_eval:
                en_f1 = lang_eval['en']['multiclass_classification']['macro_f1']
                cn_f1 = lang_eval['cn']['multiclass_classification']['macro_f1']
                diff = abs(en_f1 - cn_f1)
                
                summary['cross_lingual_degradation'] = float(diff)
                summary['better_language'] = 'English' if en_f1 > cn_f1 else 'Chinese'
        
        return summary
    
    def _print_binary_results(self, metrics: Dict):
        """Print binary classification results"""
        print(f"\nBinary Classification Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  MCC: {metrics['matthews_corrcoef']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TN: {cm['tn']} | FP: {cm['fp']}")
        print(f"  FN: {cm['fn']} | TP: {cm['tp']}")
    
    def _print_multiclass_results(self, metrics: Dict, task_name: str):
        """Print multiclass results"""
        print(f"\n{task_name} Results:")
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        
        print(f"\nAverage Metrics:")
        print(f"  Macro - Precision: {metrics['macro']['precision']:.4f}, "
              f"Recall: {metrics['macro']['recall']:.4f}, "
              f"F1: {metrics['macro']['f1_score']:.4f}")
        
        if metrics['per_class']:
            print(f"\nPer-class Metrics:")
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"  {class_name}:")
                print(f"    P: {class_metrics['precision']:.4f}, "
                      f"R: {class_metrics['recall']:.4f}, "
                      f"F1: {class_metrics['f1_score']:.4f}, "
                      f"Support: {class_metrics['support']}")
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file"""
        logger.info(f"Saving results to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def generate_report(self, results: Dict, output_file: str):
        """Generate text format evaluation report"""
        logger.info(f"Generating report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Medical Misinformation Detection - Evaluation Report\n")
            f.write("="*80 + "\n\n")
            
            # Overall performance
            f.write("Overall Performance Metrics:\n")
            summary = results.get('summary', {}).get('overall_performance', {})
            f.write(f"  Binary F1: {summary.get('binary_f1', 0):.4f}\n")
            f.write(f"  Binary MCC: {summary.get('binary_mcc', 0):.4f}\n")
            f.write(f"  5-class Macro F1: {summary.get('multiclass_macro_f1', 0):.4f}\n")
            f.write(f"  5-class Accuracy: {summary.get('multiclass_accuracy', 0):.4f}\n\n")
            
            # Language comparison if available
            if 'language_evaluation' in results:
                f.write("Cross-lingual Performance:\n")
                for lang_code, metrics in results['language_evaluation'].items():
                    f.write(f"\n{metrics['language_name']} ({metrics['n_samples']} samples):\n")
                    f.write(f"  Binary F1: {metrics['binary_classification']['f1_score']:.4f}\n")
                    f.write(f"  Multi Macro F1: {metrics['multiclass_classification']['macro_f1']:.4f}\n")
                
                if 'cross_lingual_degradation' in results['summary']:
                    f.write(f"\nCross-lingual Degradation: {results['summary']['cross_lingual_degradation']:.4f}\n")
    
    def visualize_results(self, results: Dict, output_prefix: str = "evaluation"):
        """Generate visualization charts"""
        logger.info(f"Generating visualizations: {output_prefix}_*.png")
        
        # Setup plot style
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Binary confusion matrix
        binary = results['binary_classification']
        cm_binary = np.array([
            [binary['confusion_matrix']['tn'], binary['confusion_matrix']['fp']],
            [binary['confusion_matrix']['fn'], binary['confusion_matrix']['tp']]
        ])
        
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Error', 'Has Error'],
                   yticklabels=['No Error', 'Has Error'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Binary Classification Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. 5-class confusion matrix
        full_results = results['5_class_classification']
        cm_full = np.array(full_results['confusion_matrix'])
        
        english_labels = ['none', 'over_gen', 'improper', 'exagger', 'spurious']
        
        sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens',
                   xticklabels=english_labels[:cm_full.shape[1]],
                   yticklabels=english_labels[:cm_full.shape[0]],
                   ax=axes[0, 1])
        axes[0, 1].set_title('5-class Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. F1 score comparison
        f1_scores = {
            'Binary': results['binary_classification']['f1_score'],
            '4-class': results['4_class_classification']['macro']['f1_score'],
            '5-class': results['5_class_classification']['macro']['f1_score']
        }
        
        axes[1, 0].bar(f1_scores.keys(), f1_scores.values(), color=['blue', 'orange', 'green'])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        
        for i, (k, v) in enumerate(f1_scores.items()):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 4. Per-class F1 scores
        per_class = full_results['per_class']
        class_names = list(per_class.keys())
        f1_values = [per_class[c]['f1_score'] for c in class_names]
        
        axes[1, 1].bar(range(len(class_names)), f1_values)
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels(english_labels[:len(class_names)], rotation=45)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Per-class F1 Scores')
        axes[1, 1].set_ylabel('F1 Score')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {output_prefix}_visualization.png")
        
        # Generate language comparison visualization if available
        if 'language_evaluation' in results:
            self._visualize_language_comparison(results, output_prefix)
    
    def _visualize_language_comparison(self, results: Dict, output_prefix: str):
        """Generate language comparison visualization"""
        
        lang_eval = results['language_evaluation']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        languages = []
        binary_f1s = []
        multi_f1s = []
        
        for lang_code, metrics in lang_eval.items():
            languages.append(metrics['language_name'])
            binary_f1s.append(metrics['binary_classification']['f1_score'])
            multi_f1s.append(metrics['multiclass_classification']['macro_f1'])
        
        # Binary F1 comparison
        x = np.arange(len(languages))
        width = 0.35
        
        axes[0].bar(x, binary_f1s, width, label='Binary F1', color='steelblue')
        axes[0].set_xlabel('Language')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('Binary Classification F1 by Language')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(languages)
        axes[0].set_ylim(0, 1)
        
        # Multi F1 comparison
        axes[1].bar(x, multi_f1s, width, label='Macro F1', color='coral')
        axes[1].set_xlabel('Language')
        axes[1].set_ylabel('Macro F1 Score')
        axes[1].set_title('5-class Macro F1 by Language')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(languages)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_language_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Language comparison saved: {output_prefix}_language_comparison.png")


def compare_models(prediction_files: List[str], output_file: str = "model_comparison.csv", 
                  include_language: bool = True):
    """Compare performance of multiple models"""
    logger.info(f"Comparing {len(prediction_files)} models")
    
    evaluator = UnifiedEvaluator()
    comparison_data = []
    language_comparison_data = []
    
    for pred_file in prediction_files:
        # Extract model name
        model_name = Path(pred_file).stem.replace('_predictions', '').replace('_test', '')
        
        # Load predictions
        y_true, y_pred, predictions = evaluator.load_predictions(pred_file)
        
        # Evaluate
        results = evaluator.evaluate_all(y_true, y_pred, predictions, evaluate_language=include_language)
        
        # Extract key metrics
        comparison_data.append({
            'Model': model_name,
            'Binary F1': results['binary_classification']['f1_score'],
            'Binary MCC': results['binary_classification']['matthews_corrcoef'],
            'Binary Bal-Acc': results['binary_classification']['balanced_accuracy'],
            '4-class Macro F1': results['4_class_classification']['macro']['f1_score'],
            '4-class Accuracy': results['4_class_classification']['accuracy'],
            '5-class Macro F1': results['5_class_classification']['macro']['f1_score'],
            '5-class Accuracy': results['5_class_classification']['accuracy']
        })
        
        # Language evaluation if available
        if include_language and 'language_evaluation' in results:
            for lang_code, lang_metrics in results['language_evaluation'].items():
                language_comparison_data.append({
                    'Model': model_name,
                    'Language': lang_metrics['language_name'],
                    'Samples': lang_metrics['n_samples'],
                    'Binary F1': lang_metrics['binary_classification']['f1_score'],
                    'Binary MCC': lang_metrics['binary_classification']['mcc'],
                    'Multi Macro F1': lang_metrics['multiclass_classification']['macro_f1'],
                    'Multi Accuracy': lang_metrics['multiclass_classification']['accuracy']
                })
    
    # Create comparison table
    df = pd.DataFrame(comparison_data)
    df = df.round(4)
    
    # Sort by 5-class Macro F1
    df = df.sort_values('5-class Macro F1', ascending=False)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Model comparison saved: {output_file}")
    
    # Print table
    print("\n" + "="*100)
    print("Model Performance Comparison")
    print("="*100)
    print(df.to_string(index=False))
    
    # Language comparison if available
    if language_comparison_data:
        df_lang = pd.DataFrame(language_comparison_data)
        df_lang = df_lang.round(4)
        
        lang_output_file = output_file.replace('.csv', '_language.csv')
        df_lang.to_csv(lang_output_file, index=False)
        
        print("\n" + "="*100)
        print("Cross-lingual Performance Comparison")
        print("="*100)
        print(df_lang.to_string(index=False))
        
        # Calculate language robustness
        print("\nLanguage Robustness Analysis:")
        for model in df['Model'].unique():
            model_lang_data = df_lang[df_lang['Model'] == model]
            if len(model_lang_data) >= 2:
                f1_values = model_lang_data['Multi Macro F1'].values
                degradation = max(f1_values) - min(f1_values)
                print(f"  {model}: Degradation={degradation:.4f}")
    
    return df


def generate_latex_table_data(prediction_files: List[str], output_file: str = "latex_table_data.txt"):
    """Generate LaTeX table data (values multiplied by 100)"""
    logger.info(f"Generating LaTeX table data for {len(prediction_files)} models")
    
    evaluator = UnifiedEvaluator()
    
    all_results = []
    model_names = []
    all_predictions = []
    
    for pred_file in prediction_files:
        # Extract model name
        model_name = Path(pred_file).stem.replace('_predictions', '').replace('_test', '')
        model_names.append(model_name)
        
        # Load predictions
        y_true, y_pred, predictions = evaluator.load_predictions(pred_file)
        all_predictions.append(predictions)
        
        # Evaluate
        results = evaluator.evaluate_all(y_true, y_pred, predictions)
        all_results.append(results)
    
    # Generate Table 1 data (model performance comparison)
    table1_lines = []
    
    for i, (model_name, results) in enumerate(zip(model_names, all_results)):
        binary = results['binary_classification']
        multi4 = results['4_class_classification']
        multi5 = results['5_class_classification']
        
        # Extract metrics and multiply by 100
        values = [
            f"{binary['precision']*100:.1f}",
            f"{binary['recall']*100:.1f}",
            f"{binary['f1_score']*100:.1f}",
            f"{binary['matthews_corrcoef']*100:.1f}",
            f"{binary['balanced_accuracy']*100:.1f}",
            f"{multi4['macro']['f1_score']*100:.1f}",
            f"{multi4['accuracy']*100:.1f}",
            f"{multi5['macro']['f1_score']*100:.1f}",
            f"{multi5['accuracy']*100:.1f}"
        ]
        
        table1_line = " & ".join(values)
        table1_lines.append(f"{model_name} & {table1_line} \\\\")
    
    # Generate Table 2 data (language comparison)
    table2_lines = []
    
    for i, (model_name, predictions) in enumerate(zip(model_names, all_predictions)):
        if predictions and 'language' in predictions[0]:
            languages = [p.get('language', 'unknown') for p in predictions]
            y_true = np.array([p['true_label'] for p in predictions])
            y_pred = np.array([p['pred_label'] for p in predictions])
            
            # English performance
            en_mask = np.array([lang == 'en' for lang in languages])
            cn_mask = np.array([lang == 'cn' for lang in languages])
            
            # Calculate English metrics
            if en_mask.sum() > 0:
                y_true_en = y_true[en_mask]
                y_pred_en = y_pred[en_mask]
                
                y_true_en_binary = (y_true_en > 0).astype(int)
                y_pred_en_binary = (y_pred_en > 0).astype(int)
                en_binary_f1 = f1_score(y_true_en_binary, y_pred_en_binary, zero_division=0)
                en_multi_f1 = f1_score(y_true_en, y_pred_en, average='macro', zero_division=0)
            else:
                en_binary_f1 = 0
                en_multi_f1 = 0
            
            # Calculate Chinese metrics
            if cn_mask.sum() > 0:
                y_true_cn = y_true[cn_mask]
                y_pred_cn = y_pred[cn_mask]
                
                y_true_cn_binary = (y_true_cn > 0).astype(int)
                y_pred_cn_binary = (y_pred_cn > 0).astype(int)
                cn_binary_f1 = f1_score(y_true_cn_binary, y_pred_cn_binary, zero_division=0)
                cn_multi_f1 = f1_score(y_true_cn, y_pred_cn, average='macro', zero_division=0)
            else:
                cn_binary_f1 = 0
                cn_multi_f1 = 0
            
            # Calculate difference
            diff = abs(en_multi_f1 - cn_multi_f1)
            
            # Multiply by 100
            values = [
                f"{en_binary_f1*100:.1f}",
                f"{en_multi_f1*100:.1f}",
                f"{cn_binary_f1*100:.1f}",
                f"{cn_multi_f1*100:.1f}",
                f"{diff*100:.1f}"
            ]
            
            table2_line = " & ".join(values)
            table2_lines.append(f"{model_name} & {table2_line} \\\\")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Table 1: Model Performance Comparison\n")
        f.write("% Columns: Model & P & R & F1 & MCC & Bal-Acc & 4-class F1 & 4-class Acc & 5-class F1 & 5-class Acc\n")
        f.write("% All values multiplied by 100\n")
        for line in table1_lines:
            f.write(line + "\n")
        
        if table2_lines:
            f.write("\n% Table 2: Cross-lingual Performance Comparison\n")
            f.write("% Columns: Model & EN F1(Binary) & EN F1(5-class) & CN F1(Binary) & CN F1(5-class) & Degradation\n")
            f.write("% All values multiplied by 100\n")
            for line in table2_lines:
                f.write(line + "\n")
    
    # Print to console
    print("\n" + "="*80)
    print("LaTeX Table Data (values multiplied by 100)")
    print("="*80)
    
    print("\nTable 1: Model Performance Comparison")
    print("Columns: Model & P & R & F1 & MCC & Bal-Acc & 4-class F1 & 4-class Acc & 5-class F1 & 5-class Acc")
    for line in table1_lines:
        print(line)
    
    if table2_lines:
        print("\nTable 2: Cross-lingual Performance Comparison")
        print("Columns: Model & EN F1(Binary) & EN F1(5-class) & CN F1(Binary) & CN F1(5-class) & Degradation")
        for line in table2_lines:
            print(line)
    
    logger.info(f"\nLaTeX table data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation System')
    
    parser.add_argument('--input', type=str, required=False,
                       help='Input prediction JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output evaluation JSON file')
    parser.add_argument('--report', type=str, default=None,
                       help='Generate text report')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization charts')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple model predictions')
    parser.add_argument('--latex', nargs='+',
                       help='Generate LaTeX table data')
    parser.add_argument('--language', action='store_true',
                       help='Include language evaluation')
    
    args = parser.parse_args()
    
    # LaTeX mode
    if args.latex:
        generate_latex_table_data(args.latex, "latex_table_data.txt")
        return
    
    # Comparison mode
    if args.compare:
        compare_models(args.compare, include_language=args.language)
        return
    
    # Single evaluation mode
    if not args.input:
        parser.error("--input is required unless using --compare or --latex mode")
    
    # Create evaluator
    evaluator = UnifiedEvaluator()
    
    # Load predictions
    y_true, y_pred, predictions = evaluator.load_predictions(args.input)
    
    # Evaluate
    results = evaluator.evaluate_all(y_true, y_pred, predictions, evaluate_language=args.language)
    
    # Save results
    if args.output:
        evaluator.save_results(results, args.output)
    else:
        input_path = Path(args.input)
        output_file = str(input_path.parent / f"{input_path.stem}_evaluation.json")
        evaluator.save_results(results, output_file)
    
    # Generate report
    if args.report:
        evaluator.generate_report(results, args.report)
    
    # Generate visualization
    if args.visualize:
        input_path = Path(args.input)
        output_prefix = str(input_path.parent / f"{input_path.stem}_evaluation")
        evaluator.visualize_results(results, output_prefix)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()
