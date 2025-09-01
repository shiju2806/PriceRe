"""
Test Framework for Data Cleaning Validation
Tests semantic, statistical, and hybrid cleaning approaches
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json

from .test_data_generator import TestCase, TestDataGenerator, generate_test_suite
from .semantic_detector import SemanticJunkDetector, SemanticConfig

logger = logging.getLogger(__name__)

@dataclass
class CleaningMetrics:
    """Metrics for evaluating cleaning performance"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    processing_time: float
    rows_removed: int
    false_positives: int
    false_negatives: int
    total_junk_actual: int
    total_junk_predicted: int
    confidence_distribution: Dict[str, int]

@dataclass
class TestResult:
    """Results from running a test case"""
    test_name: str
    passed: bool
    metrics: CleaningMetrics
    predicted_junk: List[int]
    actual_junk: List[int]
    error_message: Optional[str] = None
    explanation: Optional[Dict] = None

class CleaningTestFramework:
    """Framework for testing and validating cleaning algorithms"""
    
    def __init__(self):
        self.test_generator = TestDataGenerator()
        self.test_results = {}
        
    def run_test_suite(self, cleaner, test_suite: Optional[Dict[str, TestCase]] = None) -> Dict[str, TestResult]:
        """Run full test suite against a cleaning algorithm"""
        
        if test_suite is None:
            test_suite = generate_test_suite()
        
        logger.info(f"Running test suite with {len(test_suite)} test cases")
        
        results = {}
        
        for test_name, test_case in test_suite.items():
            logger.info(f"Running test: {test_name}")
            
            try:
                result = self.run_single_test(cleaner, test_case, test_name)
                results[test_name] = result
                
                status = "PASS" if result.passed else "FAIL"
                logger.info(f"Test {test_name}: {status} (F1: {result.metrics.f1_score:.3f})")
                
            except Exception as e:
                logger.error(f"Test {test_name} failed with error: {e}")
                results[test_name] = TestResult(
                    test_name=test_name,
                    passed=False,
                    metrics=CleaningMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}),
                    predicted_junk=[],
                    actual_junk=test_case.junk_row_indices,
                    error_message=str(e)
                )
        
        self.test_results = results
        return results
    
    def run_single_test(self, cleaner, test_case: TestCase, test_name: str) -> TestResult:
        """Run a single test case"""
        
        start_time = time.time()
        
        # Run cleaning algorithm
        if hasattr(cleaner, 'detect_junk_rows'):
            # Semantic detector interface
            predicted_junk = cleaner.detect_junk_rows(test_case.messy_df)
        elif hasattr(cleaner, 'clean'):
            # General cleaner interface
            cleaned_df = cleaner.clean(test_case.messy_df)
            predicted_junk = self._infer_removed_rows(test_case.messy_df, cleaned_df)
        else:
            raise ValueError("Cleaner must have either 'detect_junk_rows' or 'clean' method")
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            predicted_junk=predicted_junk,
            actual_junk=test_case.junk_row_indices,
            processing_time=processing_time
        )
        
        # Check if test passed
        passed = metrics.f1_score >= test_case.expected_accuracy
        
        return TestResult(
            test_name=test_name,
            passed=passed,
            metrics=metrics,
            predicted_junk=predicted_junk,
            actual_junk=test_case.junk_row_indices,
            explanation=self._generate_test_explanation(test_case, predicted_junk, metrics)
        )
    
    def _infer_removed_rows(self, original_df: pl.DataFrame, cleaned_df: pl.DataFrame) -> List[int]:
        """Infer which rows were removed by comparing original and cleaned dataframes"""
        
        if len(cleaned_df) >= len(original_df):
            return []  # No rows removed
        
        # Simple approach: assume removed rows are at indices that are missing
        # This is approximate and may not work for all cleaners
        original_rows = len(original_df)
        cleaned_rows = len(cleaned_df)
        removed_count = original_rows - cleaned_rows
        
        # This is a limitation - we can't easily infer which specific rows were removed
        # without more sophisticated tracking. For now, return empty list.
        # TODO: Enhance cleaners to return removed row indices
        logger.warning("Cannot infer specific removed rows - cleaner should return indices")
        return []
    
    def _calculate_metrics(self, predicted_junk: List[int], actual_junk: List[int], 
                          processing_time: float) -> CleaningMetrics:
        """Calculate comprehensive metrics for cleaning performance"""
        
        predicted_set = set(predicted_junk)
        actual_set = set(actual_junk)
        
        # Basic metrics
        true_positives = len(predicted_set & actual_set)
        false_positives = len(predicted_set - actual_set)
        false_negatives = len(actual_set - predicted_set)
        
        # Precision, Recall, F1
        precision = true_positives / len(predicted_set) if predicted_set else 0
        recall = true_positives / len(actual_set) if actual_set else 1  # Perfect if no actual junk
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy (considering both junk and non-junk correctly identified)
        # This requires knowing total rows, which we can't easily get here
        # For now, use F1 as proxy for accuracy
        accuracy = f1_score
        
        return CleaningMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            processing_time=processing_time,
            rows_removed=len(predicted_junk),
            false_positives=false_positives,
            false_negatives=false_negatives,
            total_junk_actual=len(actual_junk),
            total_junk_predicted=len(predicted_junk),
            confidence_distribution={}  # TODO: Add confidence distribution if available
        )
    
    def _generate_test_explanation(self, test_case: TestCase, predicted_junk: List[int], 
                                  metrics: CleaningMetrics) -> Dict[str, Any]:
        """Generate human-readable explanation of test results"""
        
        predicted_set = set(predicted_junk)
        actual_set = set(test_case.junk_row_indices)
        
        explanation = {
            "test_description": test_case.description,
            "expected_accuracy": test_case.expected_accuracy,
            "actual_f1": metrics.f1_score,
            "passed": metrics.f1_score >= test_case.expected_accuracy,
            "performance": {
                "processing_time": f"{metrics.processing_time:.3f}s",
                "rows_per_second": len(test_case.messy_df) / metrics.processing_time if metrics.processing_time > 0 else 0
            },
            "confusion": {
                "correct_detections": list(predicted_set & actual_set),
                "missed_junk": list(actual_set - predicted_set),
                "false_alarms": list(predicted_set - actual_set)
            }
        }
        
        # Add qualitative assessment
        if metrics.f1_score >= 0.9:
            explanation["assessment"] = "Excellent performance"
        elif metrics.f1_score >= 0.8:
            explanation["assessment"] = "Good performance"
        elif metrics.f1_score >= 0.7:
            explanation["assessment"] = "Acceptable performance"
        else:
            explanation["assessment"] = "Poor performance - needs improvement"
        
        return explanation
    
    def generate_test_report(self, results: Optional[Dict[str, TestResult]] = None) -> str:
        """Generate comprehensive test report"""
        
        if results is None:
            results = self.test_results
        
        if not results:
            return "No test results available"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA CLEANING TEST REPORT")
        report_lines.append("=" * 80)
        
        # Overall statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.passed)
        failed_tests = total_tests - passed_tests
        
        report_lines.append(f"\nOVERALL RESULTS:")
        report_lines.append(f"Total Tests: {total_tests}")
        report_lines.append(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        report_lines.append(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Performance statistics
        avg_f1 = np.mean([r.metrics.f1_score for r in results.values()])
        avg_precision = np.mean([r.metrics.precision for r in results.values()])
        avg_recall = np.mean([r.metrics.recall for r in results.values()])
        avg_time = np.mean([r.metrics.processing_time for r in results.values()])
        
        report_lines.append(f"\nPERFORMANCE METRICS:")
        report_lines.append(f"Average F1 Score: {avg_f1:.3f}")
        report_lines.append(f"Average Precision: {avg_precision:.3f}")
        report_lines.append(f"Average Recall: {avg_recall:.3f}")
        report_lines.append(f"Average Processing Time: {avg_time:.3f}s")
        
        # Individual test results
        report_lines.append(f"\nINDIVIDUAL TEST RESULTS:")
        report_lines.append("-" * 80)
        
        for test_name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            report_lines.append(f"\n{test_name}: {status}")
            report_lines.append(f"  F1 Score: {result.metrics.f1_score:.3f} (threshold: {result.test_name})")
            report_lines.append(f"  Precision: {result.metrics.precision:.3f}")
            report_lines.append(f"  Recall: {result.metrics.recall:.3f}")
            report_lines.append(f"  Processing Time: {result.metrics.processing_time:.3f}s")
            
            if result.error_message:
                report_lines.append(f"  ERROR: {result.error_message}")
            
            if result.explanation:
                report_lines.append(f"  Assessment: {result.explanation.get('assessment', 'N/A')}")
        
        # Recommendations
        report_lines.append(f"\nRECOMMENDAIONS:")
        
        if failed_tests > 0:
            worst_test = min(results.values(), key=lambda r: r.metrics.f1_score)
            report_lines.append(f"- Worst performing test: {worst_test.test_name} (F1: {worst_test.metrics.f1_score:.3f})")
        
        if avg_precision < 0.8:
            report_lines.append("- Consider increasing precision (too many false positives)")
        
        if avg_recall < 0.8:
            report_lines.append("- Consider increasing recall (missing too much junk)")
        
        if avg_time > 10:
            report_lines.append("- Consider performance optimization (slow processing)")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)
    
    def save_results(self, results: Dict[str, TestResult], output_file: Path) -> None:
        """Save test results to JSON file"""
        
        serializable_results = {}
        
        for test_name, result in results.items():
            serializable_results[test_name] = {
                "test_name": result.test_name,
                "passed": result.passed,
                "metrics": {
                    "precision": result.metrics.precision,
                    "recall": result.metrics.recall,
                    "f1_score": result.metrics.f1_score,
                    "accuracy": result.metrics.accuracy,
                    "processing_time": result.metrics.processing_time,
                    "rows_removed": result.metrics.rows_removed,
                    "false_positives": result.metrics.false_positives,
                    "false_negatives": result.metrics.false_negatives
                },
                "predicted_junk": result.predicted_junk,
                "actual_junk": result.actual_junk,
                "error_message": result.error_message,
                "explanation": result.explanation
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")

class SemanticDetectorTester:
    """Specialized tester for semantic detectors"""
    
    def __init__(self):
        self.framework = CleaningTestFramework()
    
    def test_semantic_detector_configurations(self) -> Dict[str, Dict[str, TestResult]]:
        """Test different configurations of semantic detector"""
        
        configurations = {
            'default': SemanticConfig(),
            'conservative': SemanticConfig(outlier_std_multiplier=3.0),
            'aggressive': SemanticConfig(outlier_std_multiplier=1.5),
            'clustering': SemanticConfig(outlier_threshold_method='clustering'),
            'euclidean': SemanticConfig(similarity_metric='euclidean')
        }
        
        test_suite = generate_test_suite()
        all_results = {}
        
        for config_name, config in configurations.items():
            logger.info(f"Testing configuration: {config_name}")
            
            detector = SemanticJunkDetector(config)
            if not detector.initialize():
                logger.error(f"Failed to initialize detector for config: {config_name}")
                continue
            
            results = self.framework.run_test_suite(detector, test_suite)
            all_results[config_name] = results
            
            # Print quick summary
            avg_f1 = np.mean([r.metrics.f1_score for r in results.values()])
            logger.info(f"Configuration {config_name} average F1: {avg_f1:.3f}")
        
        return all_results
    
    def find_optimal_threshold(self, test_cases: List[str] = None) -> float:
        """Find optimal outlier threshold using test cases"""
        
        if test_cases is None:
            test_cases = ['empty_rows', 'header_misplaced', 'footer_junk']
        
        test_suite = generate_test_suite()
        selected_tests = {k: v for k, v in test_suite.items() if k in test_cases}
        
        best_threshold = 2.0
        best_avg_f1 = 0
        
        for threshold in np.arange(1.0, 4.0, 0.2):
            config = SemanticConfig(outlier_std_multiplier=threshold)
            detector = SemanticJunkDetector(config)
            
            if not detector.initialize():
                continue
            
            results = self.framework.run_test_suite(detector, selected_tests)
            avg_f1 = np.mean([r.metrics.f1_score for r in results.values()])
            
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_threshold = threshold
            
            logger.info(f"Threshold {threshold:.1f}: F1 = {avg_f1:.3f}")
        
        logger.info(f"Optimal threshold: {best_threshold:.1f} (F1: {best_avg_f1:.3f})")
        return best_threshold

# Convenience functions
def quick_test_semantic_detector() -> str:
    """Quick test of semantic detector with default settings"""
    
    tester = SemanticDetectorTester()
    
    # Test with basic cases
    detector = SemanticJunkDetector()
    if not detector.initialize():
        return "Failed to initialize semantic detector"
    
    test_suite = generate_test_suite()
    basic_tests = {k: v for k, v in test_suite.items() if k in ['empty_rows', 'header_misplaced']}
    
    results = tester.framework.run_test_suite(detector, basic_tests)
    return tester.framework.generate_test_report(results)

def validate_cleaning_accuracy(cleaner, expected_min_f1: float = 0.8) -> bool:
    """Validate that a cleaner meets minimum accuracy requirements"""
    
    framework = CleaningTestFramework()
    test_suite = generate_test_suite()
    
    results = framework.run_test_suite(cleaner, test_suite)
    avg_f1 = np.mean([r.metrics.f1_score for r in results.values()])
    
    return avg_f1 >= expected_min_f1