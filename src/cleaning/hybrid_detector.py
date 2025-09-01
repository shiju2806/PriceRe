"""
Hybrid Junk Detector - Combines Semantic + Statistical + (Future LLM) Approaches
Implements the adaptive pipeline: Semantic → Statistical → LLM (complex last)
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum
import time

from .semantic_detector import SemanticJunkDetector, SemanticConfig, SemanticResult
from .statistical_analyzer import StatisticalJunkDetector, StatisticalConfig, StatisticalResult
from .header_detector import detect_and_clean_headers, HeaderDetectionResult

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Different processing modes for hybrid detection"""
    FAST = "fast"              # Only semantic
    BALANCED = "balanced"      # Semantic + Statistical
    COMPREHENSIVE = "comprehensive"  # Semantic + Statistical + LLM (future)

@dataclass
class HybridConfig:
    """Configuration for hybrid detection"""
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    semantic_weight: float = 0.4
    statistical_weight: float = 0.4
    llm_weight: float = 0.2  # For future use
    
    # Confidence thresholds for each layer
    semantic_confidence_threshold: float = 0.8
    statistical_confidence_threshold: float = 0.7
    
    # Performance optimization
    early_exit_threshold: float = 0.95  # If semantic confidence is very high, skip other layers
    max_processing_time: float = 30.0   # Maximum processing time in seconds
    
    # Layer-specific configs
    semantic_config: Optional[SemanticConfig] = None
    statistical_config: Optional[StatisticalConfig] = None

@dataclass
class HybridResult:
    """Results from hybrid detection"""
    junk_row_indices: List[int]
    final_confidence_scores: np.ndarray
    semantic_result: Optional[SemanticResult] = None
    statistical_result: Optional[StatisticalResult] = None
    llm_result: Optional[Any] = None  # Future LLM results
    header_detection_result: Optional[HeaderDetectionResult] = None  # NEW
    
    processing_mode_used: ProcessingMode = ProcessingMode.BALANCED
    layers_used: List[str] = None
    processing_time: float = 0.0
    early_exit_triggered: bool = False
    
    # Performance metrics
    semantic_processing_time: float = 0.0
    statistical_processing_time: float = 0.0
    llm_processing_time: float = 0.0
    header_detection_time: float = 0.0  # NEW

class HybridJunkDetector:
    """
    Hybrid detector combining multiple approaches
    Implements progressive enhancement: Conservative → Aggressive
    """
    
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()
        
        # Initialize detectors based on config
        self.semantic_detector = None
        self.statistical_detector = None
        self.llm_detector = None  # Future
        
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize the detection layers"""
        
        # Semantic detector (always available)
        try:
            semantic_config = self.config.semantic_config or SemanticConfig()
            self.semantic_detector = SemanticJunkDetector(semantic_config)
            logger.info("Semantic detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic detector: {e}")
            self.semantic_detector = None
        
        # Statistical detector (always available)
        statistical_config = self.config.statistical_config or StatisticalConfig()
        self.statistical_detector = StatisticalJunkDetector(statistical_config)
        logger.info("Statistical detector initialized")
        
        # LLM detector (future implementation)
        # self.llm_detector = LLMJunkDetector() # TODO: Phase 3
    
    def detect_junk_rows(self, df: pl.DataFrame, 
                        return_detailed: bool = False) -> Union[List[int], HybridResult]:
        """
        Detect junk rows using hybrid approach with smart header detection
        
        Args:
            df: Input dataframe (raw, no assumptions about headers)
            return_detailed: Return detailed HybridResult if True
            
        Returns:
            List of junk row indices or detailed HybridResult
        """
        start_time = time.time()
        
        logger.info(f"Starting hybrid junk detection on {df.shape[0]} rows using {self.config.processing_mode.value} mode")
        
        # Step 0: Smart Header Detection (NEW)
        header_start = time.time()
        cleaned_df, header_result = detect_and_clean_headers(df, min_confidence=0.6)
        header_time = time.time() - header_start
        
        logger.info(f"Header detection: found header at row {header_result.header_row_index}, "
                   f"removed {len(header_result.rows_to_remove_above)} rows above")
        
        # Continue with cleaned dataframe
        working_df = cleaned_df
        
        result = HybridResult(
            junk_row_indices=[],
            final_confidence_scores=np.zeros(working_df.shape[0]),
            processing_mode_used=self.config.processing_mode,
            layers_used=[],
            header_detection_result=header_result,
            header_detection_time=header_time
        )
        
        try:
            # Layer 1: Semantic Detection
            if self.config.processing_mode != ProcessingMode.FAST or self.semantic_detector:
                result = self._run_semantic_layer(working_df, result)
                
                # Check for early exit
                if self._should_exit_early(result):
                    result.early_exit_triggered = True
                    logger.info("Early exit triggered after semantic layer")
                else:
                    # Layer 2: Statistical Enhancement  
                    if self.config.processing_mode in [ProcessingMode.BALANCED, ProcessingMode.COMPREHENSIVE]:
                        result = self._run_statistical_layer(working_df, result)
                    
                    # Layer 3: LLM Enhancement (future)
                    if self.config.processing_mode == ProcessingMode.COMPREHENSIVE:
                        result = self._run_llm_layer(working_df, result)
            
            # Fallback to statistical only if semantic failed
            elif self.statistical_detector:
                logger.warning("Semantic detector unavailable, using statistical only")
                result = self._run_statistical_layer(working_df, result)
            
            else:
                logger.error("No detection layers available")
                raise RuntimeError("No detection layers available")
            
            # Finalize results
            result = self._finalize_results(result)
            
        except Exception as e:
            logger.error(f"Hybrid detection failed: {e}")
            # Fallback to simple statistical detection
            result = self._fallback_detection(working_df, result)
        
        result.processing_time = time.time() - start_time
        logger.info(f"Hybrid detection completed in {result.processing_time:.2f}s, found {len(result.junk_row_indices)} junk rows")
        
        if return_detailed:
            return result
        else:
            return result.junk_row_indices
    
    def _run_semantic_layer(self, df: pl.DataFrame, result: HybridResult) -> HybridResult:
        """Run semantic detection layer"""
        
        if not self.semantic_detector:
            logger.warning("Semantic detector not available")
            return result
        
        try:
            semantic_start = time.time()
            
            # Initialize if needed
            if not self.semantic_detector.is_initialized:
                if not self.semantic_detector.initialize():
                    logger.error("Failed to initialize semantic detector")
                    return result
            
            # Run semantic detection
            semantic_result = self.semantic_detector.detect_junk_rows(df, return_detailed=True)
            
            result.semantic_result = semantic_result
            result.semantic_processing_time = time.time() - semantic_start
            result.layers_used.append("semantic")
            
            # Update confidence scores
            result.final_confidence_scores = semantic_result.confidence_scores * self.config.semantic_weight
            
            logger.info(f"Semantic layer completed: {len(semantic_result.junk_row_indices)} junk rows detected")
            
        except Exception as e:
            logger.error(f"Semantic layer failed: {e}")
        
        return result
    
    def _run_statistical_layer(self, df: pl.DataFrame, result: HybridResult) -> HybridResult:
        """Run statistical detection layer"""
        
        try:
            statistical_start = time.time()
            
            # Run statistical detection
            statistical_result = self.statistical_detector.detect_junk_rows(df, return_detailed=True)
            
            result.statistical_result = statistical_result
            result.statistical_processing_time = time.time() - statistical_start
            result.layers_used.append("statistical")
            
            # Combine with existing confidence scores
            statistical_scores = statistical_result.combined_scores * self.config.statistical_weight
            
            if result.semantic_result:
                # Weighted combination
                result.final_confidence_scores = (
                    result.final_confidence_scores + statistical_scores
                )
            else:
                # Statistical only
                result.final_confidence_scores = statistical_scores
            
            logger.info(f"Statistical layer completed: {len(statistical_result.junk_row_indices)} junk rows detected")
            
        except Exception as e:
            logger.error(f"Statistical layer failed: {e}")
        
        return result
    
    def _run_llm_layer(self, df: pl.DataFrame, result: HybridResult) -> HybridResult:
        """Run LLM detection layer (future implementation)"""
        
        logger.info("LLM layer not yet implemented - Phase 3 feature")
        
        # TODO: Implement LLM-based detection
        # This would use local Ollama for complex pattern recognition
        # - Context understanding
        # - Domain-specific knowledge  
        # - Complex reasoning about data relationships
        
        return result
    
    def _should_exit_early(self, result: HybridResult) -> bool:
        """Check if we should exit early based on confidence"""
        
        if not result.semantic_result:
            return False
        
        # Check if semantic confidence is very high
        high_confidence_ratio = np.mean(
            result.semantic_result.confidence_scores > self.config.early_exit_threshold
        )
        
        return high_confidence_ratio > 0.8  # 80% of rows have very high confidence
    
    def _finalize_results(self, result: HybridResult) -> HybridResult:
        """Finalize the hybrid results with improved combination logic"""
        
        if len(result.layers_used) > 1:
            # Multiple layers - use hybrid approach
            junk_indices_set = set()
            
            # Collect high-confidence detections from each layer
            if result.semantic_result:
                # Add semantic detections
                semantic_indices = set(result.semantic_result.junk_row_indices)
                junk_indices_set.update(semantic_indices)
                
            if result.statistical_result:
                # Add statistical detections
                statistical_indices = set(result.statistical_result.junk_row_indices)
                junk_indices_set.update(statistical_indices)
                
                # Also use combined confidence scores with lower threshold for edge cases
                if len(result.final_confidence_scores) > 0:
                    # More aggressive threshold for hybrid detection
                    combined_threshold = self._calculate_dynamic_threshold(result.final_confidence_scores, aggressive=True)
                    combined_detections = np.where(result.final_confidence_scores > combined_threshold)[0].tolist()
                    junk_indices_set.update(combined_detections)
            
            result.junk_row_indices = sorted(list(junk_indices_set))
            
        elif result.semantic_result:
            # Semantic only
            result.junk_row_indices = result.semantic_result.junk_row_indices
        elif result.statistical_result:
            # Statistical only
            result.junk_row_indices = result.statistical_result.junk_row_indices
        else:
            # No valid results
            result.junk_row_indices = []
        
        return result
    
    def _calculate_dynamic_threshold(self, scores: np.ndarray, aggressive: bool = False) -> float:
        """Calculate dynamic threshold for combined scores"""
        
        if len(scores) == 0:
            return 0.5
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if aggressive:
            # More aggressive threshold for better junk detection
            threshold = mean_score + (0.8 * std_score)  # Lower multiplier
            threshold = max(0.2, min(0.6, threshold))   # Lower bounds
        else:
            # Conservative threshold for combined scores
            threshold = mean_score + (1.5 * std_score)
            threshold = max(0.3, min(0.8, threshold))
        
        return threshold
    
    def _fallback_detection(self, df: pl.DataFrame, result: HybridResult) -> HybridResult:
        """Fallback detection when main methods fail"""
        
        logger.warning("Using fallback detection")
        
        try:
            # Simple rule-based fallback
            junk_indices = []
            
            for i in range(df.shape[0]):
                row = df[i, :]
                
                # Check for completely empty rows
                non_null_count = sum(1 for val in row if val is not None and str(val).strip())
                if non_null_count == 0:
                    junk_indices.append(i)
                
                # Check for rows with common junk keywords
                row_text = ' '.join(str(val) for val in row if val is not None).lower()
                junk_keywords = ['total', 'sum', 'report', 'page', 'sheet', 'header', 'footer']
                if any(keyword in row_text for keyword in junk_keywords):
                    junk_indices.append(i)
            
            result.junk_row_indices = junk_indices
            result.layers_used = ["fallback"]
            
        except Exception as e:
            logger.error(f"Even fallback detection failed: {e}")
            result.junk_row_indices = []
        
        return result
    
    def get_detection_explanation(self, df: pl.DataFrame, result: HybridResult, 
                                row_index: int) -> Dict[str, Any]:
        """Get comprehensive explanation for why a row was flagged"""
        
        if row_index not in result.junk_row_indices:
            return {"is_junk": False, "explanation": "Row was not flagged as junk"}
        
        explanation = {
            "is_junk": True,
            "row_index": row_index,
            "processing_mode": result.processing_mode_used.value,
            "layers_used": result.layers_used,
            "final_confidence": float(result.final_confidence_scores[row_index]),
            "layer_details": {}
        }
        
        # Add semantic explanation if available
        if result.semantic_result and hasattr(result.semantic_result, 'similarity_matrix'):
            try:
                semantic_explanation = self.semantic_detector.get_outlier_explanation(
                    result.semantic_result, df, row_index
                )
                explanation["layer_details"]["semantic"] = semantic_explanation
            except Exception as e:
                logger.warning(f"Failed to get semantic explanation: {e}")
        
        # Add statistical explanation if available
        if result.statistical_result:
            explanation["layer_details"]["statistical"] = {
                "content_score": float(result.statistical_result.content_scores[row_index]),
                "proximity_score": float(result.statistical_result.proximity_scores[row_index]),
                "combined_score": float(result.statistical_result.combined_scores[row_index]),
                "feature_importance": result.statistical_result.feature_importance
            }
        
        return explanation
    
    def auto_tune_thresholds(self, df: pl.DataFrame, known_junk_indices: List[int] = None) -> Dict[str, float]:
        """Auto-tune detection thresholds based on sample data"""
        
        if not known_junk_indices:
            logger.warning("No ground truth provided for threshold tuning")
            return {}
        
        logger.info(f"Auto-tuning thresholds with {len(known_junk_indices)} known junk rows")
        
        best_thresholds = {}
        
        # Test different weight combinations
        best_f1 = 0
        best_config = None
        
        for semantic_weight in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for statistical_weight in [0.2, 0.3, 0.4, 0.5, 0.6]:
                if semantic_weight + statistical_weight > 1.0:
                    continue
                
                # Temporarily adjust weights
                old_semantic = self.config.semantic_weight
                old_statistical = self.config.statistical_weight
                
                self.config.semantic_weight = semantic_weight
                self.config.statistical_weight = statistical_weight
                
                try:
                    # Run detection
                    detected_indices = self.detect_junk_rows(df)
                    
                    # Calculate F1 score
                    predicted_set = set(detected_indices)
                    actual_set = set(known_junk_indices)
                    
                    if len(predicted_set) == 0:
                        continue
                    
                    true_positives = len(predicted_set & actual_set)
                    precision = true_positives / len(predicted_set)
                    recall = true_positives / len(actual_set) if len(actual_set) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = {
                            'semantic_weight': semantic_weight,
                            'statistical_weight': statistical_weight,
                            'f1_score': f1
                        }
                
                except Exception as e:
                    logger.warning(f"Tuning failed for weights {semantic_weight}, {statistical_weight}: {e}")
                
                finally:
                    # Restore original weights
                    self.config.semantic_weight = old_semantic
                    self.config.statistical_weight = old_statistical
        
        if best_config:
            logger.info(f"Best configuration found: {best_config}")
            best_thresholds = best_config
        
        return best_thresholds

# Convenience functions
def create_hybrid_detector(processing_mode: str = "balanced",
                          semantic_weight: float = 0.4,
                          statistical_weight: float = 0.4) -> HybridJunkDetector:
    """Create a configured hybrid detector"""
    
    mode_map = {
        "fast": ProcessingMode.FAST,
        "balanced": ProcessingMode.BALANCED, 
        "comprehensive": ProcessingMode.COMPREHENSIVE
    }
    
    config = HybridConfig(
        processing_mode=mode_map.get(processing_mode, ProcessingMode.BALANCED),
        semantic_weight=semantic_weight,
        statistical_weight=statistical_weight,
        llm_weight=1.0 - semantic_weight - statistical_weight
    )
    
    return HybridJunkDetector(config)

def quick_hybrid_clean(df: pl.DataFrame, 
                      processing_mode: str = "balanced") -> Tuple[pl.DataFrame, List[int], HeaderDetectionResult]:
    """Quick hybrid cleaning with smart header detection"""
    
    detector = create_hybrid_detector(processing_mode)
    result = detector.detect_junk_rows(df, return_detailed=True)
    
    # Get the cleaned dataframe after header detection
    if result.header_detection_result:
        # Start with dataframe after header cleaning
        cleaned_df, header_result = detect_and_clean_headers(df)
        
        # Apply additional junk row removal
        if result.junk_row_indices:
            mask = pl.Series(range(cleaned_df.shape[0])).is_in(result.junk_row_indices).not_()
            final_df = cleaned_df.filter(mask)
        else:
            final_df = cleaned_df
        
        # Return total junk removed (header rows + additional junk)
        total_junk_removed = header_result.rows_to_remove_above + result.junk_row_indices
        
        return final_df, total_junk_removed, header_result
    else:
        # Fallback to original approach
        if result.junk_row_indices:
            mask = pl.Series(range(df.shape[0])).is_in(result.junk_row_indices).not_()
            clean_df = df.filter(mask)
        else:
            clean_df = df
        
        return clean_df, result.junk_row_indices, None