"""
Semantic Junk Detector - Core Phase 1 Implementation
Uses sentence transformers + vector similarity to detect outlier rows without hard-coding
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from enum import Enum

# Import semantic libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SimilarityMetric(Enum):
    """Supported similarity metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean" 
    MANHATTAN = "manhattan"

@dataclass
class SemanticConfig:
    """Configuration for semantic detection"""
    model_name: str = 'all-MiniLM-L6-v2'  # Fast, good quality model
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    outlier_threshold_method: str = 'statistical'  # 'statistical' or 'clustering'
    outlier_std_multiplier: float = 1.5  # More aggressive (was 2.0)
    clustering_eps: float = 0.3  # For DBSCAN clustering
    min_cluster_size: int = 3
    cache_embeddings: bool = True
    batch_size: int = 32
    
    # Content-aware enhancement settings
    content_weight: float = 0.7  # Weight for content similarity
    structural_weight: float = 0.3  # Weight for structural similarity
    use_field_level_analysis: bool = True  # Analyze individual fields, not just rows
    junk_context_boost: float = 0.3  # Boost score for junk-context words

@dataclass
class SemanticResult:
    """Results from semantic detection"""
    junk_row_indices: List[int]
    outlier_scores: np.ndarray
    similarity_matrix: Optional[np.ndarray]
    cluster_labels: Optional[np.ndarray]
    threshold_used: float
    method_used: str
    confidence_scores: np.ndarray
    processing_time: float

class SemanticJunkDetector:
    """
    Semantic-based junk detection using vector similarity
    No hard-coded rules - learns patterns from data itself
    """
    
    def __init__(self, config: SemanticConfig = None):
        self.config = config or SemanticConfig()
        self.model = None
        self.embeddings_cache = {}
        self.is_initialized = False
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers required for semantic detection")
    
    def initialize(self) -> bool:
        """Initialize the semantic model"""
        try:
            logger.info(f"Loading semantic model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            self.is_initialized = True
            logger.info("Semantic detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize semantic detector: {e}")
            return False
    
    def detect_junk_rows(self, df: pl.DataFrame, 
                        return_detailed: bool = False) -> Union[List[int], SemanticResult]:
        """
        Detect junk rows using semantic similarity
        
        Args:
            df: Input dataframe
            return_detailed: If True, return detailed SemanticResult
            
        Returns:
            List of junk row indices or detailed SemanticResult
        """
        import time
        start_time = time.time()
        
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Semantic detector not initialized")
        
        logger.info(f"Starting semantic junk detection on {df.shape[0]} rows")
        
        # Step 1: Convert rows to text representations
        row_texts = self._dataframe_to_text(df)
        
        # Step 2: Generate embeddings
        embeddings = self._generate_embeddings(row_texts)
        
        # Step 3: Calculate similarity and detect outliers
        result = self._detect_outliers_from_embeddings(embeddings, df.shape[0])
        
        # Step 4: Enhance with metadata-based confidence
        result = self._enhance_with_metadata(result, df)
        
        result.processing_time = time.time() - start_time
        logger.info(f"Semantic detection completed in {result.processing_time:.2f}s, found {len(result.junk_row_indices)} junk rows")
        
        if return_detailed:
            return result
        else:
            return result.junk_row_indices
    
    def _dataframe_to_text(self, df: pl.DataFrame) -> List[str]:
        """Convert DataFrame rows to text for content-aware embedding"""
        import re
        
        row_texts = []
        
        # Analyze document structure for context
        total_rows = df.shape[0]
        
        for row_idx in range(df.shape[0]):
            row = df.row(row_idx)
            
            # Convert row to text representation
            text_parts = []
            junk_indicators = []
            
            for col_idx, (col_name, value) in enumerate(zip(df.columns, row)):
                if value is not None and str(value).strip():
                    value_str = str(value).strip()
                    
                    # Analyze for junk content
                    junk_score = self._analyze_field_junk_content(value_str, row_idx, total_rows)
                    if junk_score > 0.5:
                        junk_indicators.append(f"[JUNK_FIELD]")
                    
                    # Include column context for better semantic understanding
                    text_parts.append(f"{col_name}: {value_str}")
            
            # Create enhanced row text with context
            if text_parts:
                row_text = " | ".join(text_parts)
                
                # Add positional context
                position_ratio = row_idx / total_rows
                if position_ratio > 0.8:  # Footer area
                    row_text = f"[FOOTER_AREA] {row_text}"
                elif position_ratio < 0.2:  # Header area  
                    row_text = f"[HEADER_AREA] {row_text}"
                
                # Add junk context if detected
                if junk_indicators:
                    row_text = f"[CONTAINS_JUNK] {row_text}"
                    
            else:
                row_text = "[EMPTY_ROW]"
            
            row_texts.append(row_text)
        
        return row_texts
    
    def _analyze_field_junk_content(self, field_value: str, row_idx: int, total_rows: int) -> float:
        """Analyze individual field for junk content indicators"""
        import re
        
        value_lower = field_value.lower()
        junk_score = 0.0
        
        # Strong junk keywords
        junk_keywords = ['total', 'sum', 'summary', 'generated', 'exported', 'created', 
                        'end', 'footer', 'report', 'final', 'grand', 'system']
        
        # Check for junk keywords
        for keyword in junk_keywords:
            if keyword in value_lower:
                junk_score += 0.3
        
        # Contextual patterns that suggest junk
        junk_patterns = [
            (r'\d+\s+(products|items|records|total)', 0.8),
            (r'(total|final|grand)\s+(summary|total)', 0.9),
            (r'(generated|created|exported)\s+on', 0.8),
            (r'end\s+of\s+(report|data)', 0.9)
        ]
        
        for pattern, confidence in junk_patterns:
            if re.search(pattern, value_lower):
                junk_score += confidence
        
        # Position-based adjustment (footer area more likely junk)
        position_ratio = row_idx / total_rows
        if position_ratio > 0.8:  # Footer area
            junk_score *= 1.5
        
        return min(junk_score, 1.0)
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings for text list"""
        
        # Check cache first
        cache_key = hash(tuple(texts))
        if self.config.cache_embeddings and cache_key in self.embeddings_cache:
            logger.debug("Using cached embeddings")
            return self.embeddings_cache[cache_key]
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings in batches for memory efficiency
        embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        # Cache if enabled
        if self.config.cache_embeddings:
            self.embeddings_cache[cache_key] = embeddings
        
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _detect_outliers_from_embeddings(self, embeddings: np.ndarray, num_rows: int) -> SemanticResult:
        """Detect outlier rows from embeddings using various methods"""
        
        if self.config.outlier_threshold_method == 'clustering':
            return self._detect_outliers_clustering(embeddings, num_rows)
        else:
            return self._detect_outliers_statistical(embeddings, num_rows)
    
    def _detect_outliers_statistical(self, embeddings: np.ndarray, num_rows: int) -> SemanticResult:
        """Statistical outlier detection using similarity distances"""
        
        logger.debug("Using statistical outlier detection")
        
        # Calculate pairwise similarities
        if self.config.similarity_metric == SimilarityMetric.COSINE:
            similarity_matrix = cosine_similarity(embeddings)
        elif self.config.similarity_metric == SimilarityMetric.EUCLIDEAN:
            distance_matrix = euclidean_distances(embeddings)
            # Convert distances to similarities (inverse)
            similarity_matrix = 1 / (1 + distance_matrix)
        else:
            # Default to cosine
            similarity_matrix = cosine_similarity(embeddings)
        
        # For each row, calculate average similarity to all other rows
        outlier_scores = []
        
        for i in range(num_rows):
            # Get similarities to all other rows (excluding self)
            row_similarities = similarity_matrix[i]
            other_similarities = np.concatenate([row_similarities[:i], row_similarities[i+1:]])
            
            # Outlier score = 1 - average similarity (higher = more outlier-like)
            avg_similarity = np.mean(other_similarities)
            outlier_score = 1 - avg_similarity
            outlier_scores.append(outlier_score)
        
        outlier_scores = np.array(outlier_scores)
        
        # Determine threshold using statistical method
        mean_score = np.mean(outlier_scores)
        std_score = np.std(outlier_scores)
        threshold = mean_score + (self.config.outlier_std_multiplier * std_score)
        
        # Identify junk rows
        junk_indices = np.where(outlier_scores > threshold)[0].tolist()
        
        # Calculate confidence scores
        confidence_scores = (outlier_scores - mean_score) / (std_score + 1e-8)
        confidence_scores = np.clip(confidence_scores, 0, 1)
        
        return SemanticResult(
            junk_row_indices=junk_indices,
            outlier_scores=outlier_scores,
            similarity_matrix=similarity_matrix,
            cluster_labels=None,
            threshold_used=threshold,
            method_used='statistical_similarity',
            confidence_scores=confidence_scores,
            processing_time=0.0  # Will be set later
        )
    
    def _detect_outliers_clustering(self, embeddings: np.ndarray, num_rows: int) -> SemanticResult:
        """Clustering-based outlier detection using DBSCAN"""
        
        logger.debug("Using clustering-based outlier detection")
        
        # Use DBSCAN to find clusters
        clustering = DBSCAN(
            eps=self.config.clustering_eps,
            min_samples=self.config.min_cluster_size,
            metric='cosine'
        ).fit(embeddings)
        
        cluster_labels = clustering.labels_
        
        # Points labeled as -1 are outliers
        outlier_mask = cluster_labels == -1
        junk_indices = np.where(outlier_mask)[0].tolist()
        
        # Calculate outlier scores based on cluster membership
        outlier_scores = np.zeros(num_rows)
        
        # Outliers get score of 1.0
        outlier_scores[outlier_mask] = 1.0
        
        # For clustered points, score based on distance to cluster center
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip outliers
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_points = embeddings[cluster_mask]
            cluster_center = np.mean(cluster_points, axis=0)
            
            # Calculate distances to center
            for i, point in enumerate(embeddings):
                if cluster_labels[i] == cluster_id:
                    distance = np.linalg.norm(point - cluster_center)
                    # Normalize distance to [0, 1] range
                    outlier_scores[i] = min(distance / 2.0, 1.0)
        
        # Calculate similarity matrix for consistency
        similarity_matrix = cosine_similarity(embeddings)
        
        # Confidence scores based on cluster membership
        confidence_scores = np.where(outlier_mask, 1.0, outlier_scores)
        
        return SemanticResult(
            junk_row_indices=junk_indices,
            outlier_scores=outlier_scores,
            similarity_matrix=similarity_matrix,
            cluster_labels=cluster_labels,
            threshold_used=self.config.clustering_eps,
            method_used='clustering_dbscan',
            confidence_scores=confidence_scores,
            processing_time=0.0
        )
    
    def _enhance_with_metadata(self, result: SemanticResult, df: pl.DataFrame) -> SemanticResult:
        """Enhance semantic results with metadata-based confidence adjustments"""
        
        # Adjust confidence based on structural patterns
        enhanced_confidence = result.confidence_scores.copy()
        
        for i in range(df.shape[0]):
            row = df.row(i)
            
            # Boost confidence for obviously empty rows
            non_null_count = sum(1 for val in row if val is not None and str(val).strip())
            if non_null_count == 0:
                enhanced_confidence[i] = 1.0  # Maximum confidence for empty rows
            elif non_null_count == 1:
                enhanced_confidence[i] = min(1.0, enhanced_confidence[i] + 0.3)
            
            # Boost confidence for rows at file boundaries (likely headers/footers)
            if i < 3 or i >= len(df) - 3:  # First 3 or last 3 rows
                if i in result.junk_row_indices:
                    enhanced_confidence[i] = min(1.0, enhanced_confidence[i] + 0.2)
        
        # Update the result
        result.confidence_scores = enhanced_confidence
        
        # Re-filter junk indices based on enhanced confidence (optional)
        # Could adjust threshold here if needed
        
        return result
    
    def get_outlier_explanation(self, result: SemanticResult, df: pl.DataFrame, 
                               row_index: int) -> Dict[str, Any]:
        """Get human-readable explanation for why a row was flagged as junk"""
        
        if row_index not in result.junk_row_indices:
            return {"is_junk": False, "explanation": "Row was not flagged as junk"}
        
        row = df.row(row_index)
        row_text = self._dataframe_to_text(df.slice(row_index, 1))[0]
        
        explanation = {
            "is_junk": True,
            "row_index": row_index,
            "outlier_score": float(result.outlier_scores[row_index]),
            "confidence": float(result.confidence_scores[row_index]),
            "method": result.method_used,
            "row_content": row_text,
            "reasons": []
        }
        
        # Add specific reasons based on analysis
        if result.outlier_scores[row_index] > 0.8:
            explanation["reasons"].append("Very dissimilar to other rows in the dataset")
        
        non_null_count = sum(1 for val in row if val is not None and str(val).strip())
        if non_null_count == 0:
            explanation["reasons"].append("Completely empty row")
        elif non_null_count <= 2:
            explanation["reasons"].append("Mostly empty with very few values")
        
        if row_index < 3:
            explanation["reasons"].append("Located at the beginning of file (potential header/title)")
        elif row_index >= len(df) - 3:
            explanation["reasons"].append("Located at the end of file (potential footer/summary)")
        
        if result.cluster_labels is not None and result.cluster_labels[row_index] == -1:
            explanation["reasons"].append("Does not belong to any data cluster")
        
        return explanation
    
    def auto_tune_threshold(self, df: pl.DataFrame, known_junk_indices: List[int] = None) -> float:
        """Auto-tune the outlier threshold based on known examples (if available)"""
        
        if not known_junk_indices:
            # Use current threshold
            return self.config.outlier_std_multiplier
        
        # Generate embeddings and scores
        row_texts = self._dataframe_to_text(df)
        embeddings = self._generate_embeddings(row_texts)
        
        # Try different thresholds
        best_threshold = self.config.outlier_std_multiplier
        best_f1 = 0
        
        for threshold_mult in np.arange(1.0, 4.0, 0.2):
            # Temporarily set threshold
            old_threshold = self.config.outlier_std_multiplier
            self.config.outlier_std_multiplier = threshold_mult
            
            # Detect with this threshold
            result = self._detect_outliers_statistical(embeddings, df.shape[0])
            
            # Calculate F1 score
            predicted_junk = set(result.junk_row_indices)
            actual_junk = set(known_junk_indices)
            
            if len(predicted_junk) == 0:
                precision = recall = f1 = 0
            else:
                true_positives = len(predicted_junk & actual_junk)
                precision = true_positives / len(predicted_junk)
                recall = true_positives / len(actual_junk) if len(actual_junk) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold_mult
            
            # Restore threshold
            self.config.outlier_std_multiplier = old_threshold
        
        logger.info(f"Auto-tuned threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")
        return best_threshold

# Convenience functions
def create_semantic_detector(model_name: str = 'all-MiniLM-L6-v2', 
                           outlier_method: str = 'statistical') -> SemanticJunkDetector:
    """Create a configured semantic detector"""
    
    config = SemanticConfig(
        model_name=model_name,
        outlier_threshold_method=outlier_method
    )
    
    detector = SemanticJunkDetector(config)
    detector.initialize()
    return detector

def quick_semantic_clean(df: pl.DataFrame, 
                        model_name: str = 'all-MiniLM-L6-v2') -> Tuple[pl.DataFrame, List[int]]:
    """Quick semantic cleaning with default settings"""
    
    detector = create_semantic_detector(model_name)
    junk_indices = detector.detect_junk_rows(df)
    
    # Remove junk rows
    if junk_indices:
        mask = pl.Series(range(df.shape[0])).is_in(junk_indices).not_()
        clean_df = df.filter(mask)
    else:
        clean_df = df
    
    return clean_df, junk_indices